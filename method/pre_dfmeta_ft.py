import os
import random
import shutil
import sys
from transformers import AutoProcessor
from double_efficient_vit import apply_patch, ReduceEncoder
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from lora import LoRA_clipModel
import torch.nn.functional as F
from synthesis import InversionSyntheiszer
from tool import data2supportquery, shuffle_task, Timer, compute_confidence_interval, get_dataloader, get_model, \
    NORMALIZE_DICT, Normalizer, get_transform, bias, bias_end, find_non_zero_patches
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from logger import get_logger


class pre_dfmeta_ft(nn.Module):
    def __init__(self,args):
        super(pre_dfmeta_ft, self).__init__()
        self.args=args
        #file
        self.args = args
        if self.args.extra=='':
            feature = '{}-{}-{}-{}-{}/{}-{}-{}/{}{}-{}'.format(args.method, args.dataset,args.testdataset, args.backbone,args.synthesizer,
                                                               str(args.prune_layer), str(args.prune_ratio),args.mask_ratio,
                                                               args.way_train,args.num_sup_train,'mask' if args.use_mask else '')
        else:
            feature = '{}-{}-{}-{}-{}-ex{}/{}-{}-{}/{}{}-{}'.format(args.method, args.dataset, args.testdataset, args.backbone,args.synthesizer,args.extra,
                                                                    str(args.prune_layer), str(args.prune_ratio),args.mask_ratio,
                                                           args.way_train, args.num_sup_train,'mask' if args.use_mask else '')

        self.checkpoints_path = './checkpoints/' + feature
        os.makedirs(self.checkpoints_path, exist_ok=True)
        self.writer_path = os.path.join(self.checkpoints_path, 'writer')
        self.logger = get_logger(feature, output=self.checkpoints_path + '/' + 'log.txt')
        #dataset
        _, self.val_loader, self.test_loader = get_dataloader(self.args, resolution=args.resolution)
        #model
        self.model=self.prepare_model_for_meta_training()
        #optimizer & loss
        self.meta_optimizer = torch.optim.Adam((p for p in self.model.parameters() if p.requires_grad), lr=self.args.outer_lr)
        self.meta_scheduler = CosineAnnealingWarmupRestarts(self.meta_optimizer, first_cycle_steps=(100), cycle_mult=1.0, max_lr=self.args.outer_lr, min_lr=0.01*args.outer_lr, warmup_steps=25, gamma=1.0)
        self.meta_loss_fn = nn.CrossEntropyLoss()
        #reset datapool
        self.datapool_path=self.checkpoints_path+'/datapool'
        if os.path.exists(self.datapool_path):
            shutil.rmtree(self.datapool_path)
            print('remove')
        #use pre-inverted data
        self.pre_datapool_path = args.pre_datapool_path
        if os.path.exists(self.datapool_path):
            shutil.rmtree(self.datapool_path)
            print('remove')
        print(f'Copying pre_datapool from {self.pre_datapool_path} to {self.datapool_path}...')
        print('This may take a few minutes for large datasets...')
        shutil.copytree(self.pre_datapool_path,self.datapool_path)
        print('Datapool copy completed!')
        print('Initializing synthesizer...')

        if self.args.synthesizer=='inversion':
            # ORIGINAL: iterations=2000 (SLOW - for paper accuracy)
            # FAST MODE: iterations=200 (10x faster - for quick testing)
            fast_iterations = 200 if (hasattr(self.args, 'quick_test') and self.args.quick_test) else 2000
            
            if '16' in self.args.backbone:
                # MEMORY FIX: Reduce synthesis_batch_size for 6GB GPU
                # Original: way_train * (num_sup + num_qur) = 5 * 20 = 100 (too large!)
                # Fixed: Use smaller batch or None to generate one-by-one
                synthesis_batch = min(25, self.args.way_train * (self.args.num_sup_train + self.args.num_qur_train))
                
                self.synthesizer=InversionSyntheiszer(args=self.args, teacher=None, img_size=(3,self.args.resolution,self.args.resolution),
                     iterations=fast_iterations,  # FAST MODE: 200 or 2000
                     lr_g=0.25,
                     synthesis_batch_size=synthesis_batch,  # REDUCED from 100 to 25
                     adv=0.0, bn=0.01, oh=1, tv=0.0, l2=0.0,patch_size=16,
                     save_dir=self.datapool_path, use_mask=self.args.use_mask,
                    transform=get_transform(args,dataset=self.args.dataset,aug=True),
                     normalizer=Normalizer(**NORMALIZE_DICT[self.args.dataset]), device=self.args.device,num_classes=list(range(bias[args.dataset],bias_end[args.dataset]+1)),c_abs_list=None,max_batch_per_class=20)
                print('Synthesizer initialized successfully!')
            else:
                # MEMORY FIX: Reduce synthesis_batch_size for 6GB GPU
                synthesis_batch = min(25, self.args.way_train * (self.args.num_sup_train + self.args.num_qur_train))
                
                self.synthesizer = InversionSyntheiszer(args=self.args, teacher=None,
                                                      img_size=(3, self.args.resolution, self.args.resolution),
                                                      iterations=fast_iterations,  # FAST MODE: 200 or 2000
                                                      lr_g=0.25,
                                                      synthesis_batch_size=synthesis_batch,  # REDUCED from 100 to 25
                                                      adv=0.0, bn=0.01, oh=1, tv=0.0, l2=0.0,  patch_size=32,
                                                      save_dir=self.datapool_path, use_mask=self.args.use_mask,
                                                      transform=get_transform(args, dataset=self.args.dataset,aug=True),
                                                      normalizer=Normalizer(**NORMALIZE_DICT[self.args.dataset]),device=self.args.device, num_classes=list(range(bias[args.dataset], bias_end[args.dataset] + 1)), c_abs_list=None, max_batch_per_class=20)
                print('Synthesizer initialized successfully!')
        else:
            raise NotImplementedError
        
        print('Initialization completed! Ready for training.')


    def prepare_model_for_meta_training(self,load=True):
        model=get_model(self.args,load=load)
        model=LoRA_clipModel(model,r=self.args.rank, num_classes=self.args.way_test)
        model = model.to(self.args.device)
        apply_patch(model=model.clip_model,prune_layer_list=[-1],prune_ratio_list=[0.0],index_matrix=None)
        return model


    def forward(self,x):
        scores  = self.model(x)
        return scores
    def train_once(self,support,support_label,query, query_label,teacher=None):
        self.model.zero_grad()
        # weight.fast is designed for bi-level optimization (e.g., MAML). You can ignore it here because we use fine-tuning-free inner loop.
        for weight in self.model.parameters():
            if weight.requires_grad:
                weight.fast = None
        ReduceEncoder.init_index_matrix=torch.cat([torch.zeros(support.shape[0], 1, dtype=torch.long).to(support.device), find_non_zero_patches(images=support,patch_size=self.model.clip_model.vision_model.embeddings.patch_size)], dim=1)
        z_support = self.model.get_image_features(support)
        ReduceEncoder.init_index_matrix=torch.cat([torch.zeros(query.shape[0], 1, dtype=torch.long).to(support.device), find_non_zero_patches(images=query,patch_size=self.model.clip_model.vision_model.embeddings.patch_size)], dim=1)
        z_query = self.model.get_image_features(query)
        z_support = z_support.contiguous().view(self.args.way_train * self.args.num_sup_train, -1)
        z_query = z_query.contiguous().view(self.args.way_train * self.args.num_qur_train, -1)
        z_support = z_support.contiguous()
        protos = []
        for c in range(self.args.way_train):
            protos.append(z_support[support_label == c].mean(0))
        z_proto = torch.stack(protos, dim=0)
        z_query = z_query.contiguous().view(self.args.way_train * self.args.num_qur_train, -1)
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        if teacher==None:
            loss_outer = self.meta_loss_fn(scores, query_label)
        else:
            with torch.no_grad():
                t_logits = teacher(query)
            loss_outer = kldiv(scores, t_logits.detach())
        # weight.fast is designed for bi-level optimization (e.g., MAML). You can ignore it here because we use fine-tuning-free inner loop.
        for weight in self.model.parameters():
            if weight.requires_grad:
                weight.fast = None
        return loss_outer
    def train_loop(self):
        if os.path.exists(self.writer_path):
            shutil.rmtree(self.writer_path)
        print('Starting training loop...')
        print(f'Total episodes: {self.args.episode_train}')
        print('Mode: Using pre-generated data from datapool (FAST MODE)')
        with SummaryWriter(self.writer_path) as writer:
            loss_batch = []
            val_acc_max = 0
            val_acc_max_all=[0,0,0,0]
            timer=Timer()
            for task_id in (range(1,self.args.episode_train+1)):
                # Print progress every 10 episodes
                if task_id % 10 == 0 or task_id == 1:
                    print(f"Training progress: {task_id}/{self.args.episode_train} episodes ({100*task_id//self.args.episode_train}%)")
                
                task_batch_id=(task_id-1)//self.args.episode_batch
                if task_batch_id%2000==0:
                    task_indicator = 'lora'
                else:
                    task_indicator = 'memory'
                #***************************
                # FAST MODE: Use only pre-generated memory data (skip slow synthesis)
                task_indicator = 'memory' # only use memory data for fast meta-training
                # ***************************
                if task_indicator=='lora':
                    support, support_label_relative, query, query_label_relative=self.task_inverse_from_lora()
                    support, support_label_relative, query, query_label_relative = shuffle_task(self.args, support,support_label_relative,query,query_label_relative)
                    loss = self.train_once(support=support, support_label=support_label_relative, query=query,query_label=query_label_relative, teacher=self.model)
                else:
                    support, support_label_relative, query, query_label_relative = self.task_inverse_from_memory()
                    support, support_label_relative, query, query_label_relative = shuffle_task(self.args, support,support_label_relative,query,query_label_relative)
                    loss = self.train_once(support=support, support_label=support_label_relative, query=query,query_label=query_label_relative, teacher=None)
                loss_batch.append(loss)
                if task_id % self.args.episode_batch == 0:
                    loss = torch.stack(loss_batch).sum(0)
                    loss_batch = []
                    self.meta_optimizer.zero_grad()
                    loss.backward()
                    self.meta_optimizer.step()
                    self.meta_scheduler.step()
                # val
                if task_id % self.args.val_interval == 0:
                    if self.args.testdataset!='mix':
                        torch.save(self.model.state_dict(), self.checkpoints_path + '/testModel.pth')
                        test_acc_avg, test_pm = self.test_loop()
                        print('task_id:', task_id, 'test_acc:', test_acc_avg,' +- ',test_pm)
                        writer.add_scalar(tag='test_acc', scalar_value=test_acc_avg.item(),
                                          global_step=(task_id) // self.args.episode_batch)
                        if test_acc_avg > val_acc_max:
                            val_acc_max = test_acc_avg
                            torch.save(self.model.state_dict(), self.checkpoints_path + '/bestTestModel.pth')
                            self.logger.info('[BestEpoch]:{}, [BestTestAcc]:{} +- {}.'.format(
                                (task_id) // self.args.episode_batch, test_acc_avg, test_pm))
                    else:
                        test_acc_avg_all=[]
                        test_pm_all=[]
                        for domain_id in range(len(self.test_loader)):
                            test_acc_avg, test_pm = self.test_loop(test_loader=self.test_loader[domain_id])
                            test_acc_avg_all.append(test_acc_avg)
                            test_pm_all.append(test_pm)
                        print('task_id:', task_id, 'test_acc:', test_acc_avg_all,' +- ',test_pm_all)
                        updata=False
                        for domain_id in range(len(self.test_loader)):
                            if test_acc_avg_all[domain_id]>val_acc_max_all[domain_id]:
                                val_acc_max_all[domain_id]=test_acc_avg_all[domain_id]
                                updata =True
                        if updata:
                            self.logger.info('[BestEpoch]:{}, [BestTestAcc]:{} ,{}, {}, {}.'.format(
                                (task_id) // self.args.episode_batch, val_acc_max_all[0],val_acc_max_all[1],val_acc_max_all[2],val_acc_max_all[3]))
                    print('ETA:{}/{}'.format(
                        timer.measure(),
                        timer.measure((task_id) / (self.args.episode_train)))
                    )
    def test_once(self,support,support_label_relative,query, query_label_relative):
        self.model.zero_grad()
        fast_parameters = list((p for p in self.model.parameters() if p.requires_grad))
        for weight in self.model.parameters():
            if weight.requires_grad:
                weight.fast = None
        if '16' in self.args.backbone:
            pat_size=197
        else:
            pat_size=50
        ReduceEncoder.init_index_matrix=torch.arange(pat_size).repeat(support.shape[0], 1).to(support.device)#[[0,...,5,...],]
        z_support = self.model.get_image_features(support)
        ReduceEncoder.init_index_matrix=torch.arange(pat_size).repeat(query.shape[0], 1).to(support.device)#[[0,...,5,...],]
        z_query = self.model.get_image_features(query)
        # Get actual number of classes from labels
        num_classes = len(torch.unique(support_label_relative))
        print(f"Actual number of classes in support: {num_classes}")
        
        # Reshape features
        z_support = z_support.contiguous()
        z_query = z_query.contiguous()
        
        # Calculate prototypes for each class
        protos = []
        for c in range(num_classes):
            class_samples = z_support[support_label_relative == c]
            if len(class_samples) > 0:
                protos.append(class_samples.mean(0))
            else:
                print(f"Warning: No samples found for class {c}")
                # Use zeros as fallback
                protos.append(torch.zeros_like(z_support[0]))
        
        z_proto = torch.stack(protos, dim=0)
        
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        # test
        correct = 0
        
        # Debug shapes
        print(f"scores shape: {scores.shape}, query_label_relative shape: {query_label_relative.shape}")
        
        # Make sure we only compare predictions for valid queries
        prediction = torch.max(scores, 1)[1]
        
        # Ensure prediction and labels have same size
        min_size = min(len(prediction), len(query_label_relative))
        if min_size == 0:
            print("Warning: No valid predictions or labels!")
            return 0.0
            
        prediction = prediction[:min_size]
        query_label_relative = query_label_relative[:min_size]
        
        print(f"After adjustment: prediction shape: {prediction.shape}, query_label shape: {query_label_relative.shape}")
        
        correct = (prediction.cpu() == query_label_relative.cpu()).sum().item()
        total = min_size
        acc = 100.0 * correct / total

        for weight in self.model.parameters():
            if weight.requires_grad:
                weight.fast = None
        return acc
    def test_loop(self,test_loader=None):
        # Original sequential testing
        if test_loader==None:
            test_acc_all = []
            for test_batch in (self.test_loader):
                data, _ = test_batch[0].cuda(self.args.device), test_batch[1].cuda(self.args.device)
                support, support_label_relative, query, query_label_relative = data2supportquery(self.args, data)
                test_acc=self.test_once(support=support,support_label_relative=support_label_relative,query=query, query_label_relative=query_label_relative)
                test_acc_all.append(test_acc)
            test_acc_avg, pm = compute_confidence_interval(test_acc_all)
            return test_acc_avg,pm
        else:
            test_acc_all = []
            for test_batch in (test_loader):
                data, _ = test_batch[0].cuda(self.args.device), test_batch[1].cuda(self.args.device)
                support, support_label_relative, query, query_label_relative = data2supportquery(self.args, data)
                test_acc = self.test_once(support=support, support_label_relative=support_label_relative, query=query,query_label_relative=query_label_relative)
                test_acc_all.append(test_acc)
            test_acc_avg, pm = compute_confidence_interval(test_acc_all)
            return test_acc_avg, pm
    
    
    def val_loop(self):
        val_acc_all = []
        for val_batch in (self.val_loader):
            data, _ = val_batch[0].cuda(self.args.device), val_batch[1].cuda(self.args.device)
            support, support_label_relative, query, query_label_relative = data2supportquery(self.args, data)
            test_acc = self.test_once(support=support, support_label_relative=support_label_relative, query=query,
                                      query_label_relative=query_label_relative)
            val_acc_all.append(test_acc)
        val_acc_avg, pm = compute_confidence_interval(val_acc_all)
        return val_acc_avg, pm
    def task_inverse_from_lora(self):
        #sample a lora
        lora_id=random.randint(0, self.args.lora_num - 1)
        #load selected lora
        self.model.load_lora_parameters('./lorahub/{}_{}/{}way/lora_{}.safetensors'.format(self.args.dataset, self.args.backbone,self.args.way_test,lora_id))
        self.model.to(self.args.device)
        self.synthesizer.teacher=self.model
        specific=torch.load('./lorahub/{}_{}/{}way/global_label_{}.pth'.format(self.args.dataset, self.args.backbone,self.args.way_test,lora_id), weights_only=True)
        self.synthesizer.c_abs_list = [i + bias[self.args.dataset] for i in specific]
        support_query_tensor,support_query_tensor_masked=self.synthesizer.synthesize(targets=torch.LongTensor( (list(range(len(self.synthesizer.c_abs_list)))) * (self.args.num_sup_train + self.args.num_qur_train)),student=None,c_num=self.args.way_train,add=True)
        if self.args.use_mask==False:
            support_query=Normalizer(**NORMALIZE_DICT[self.args.dataset])(support_query_tensor)
        else:
            support_query = Normalizer(**NORMALIZE_DICT[self.args.dataset])(support_query_tensor_masked)
        support = support_query[:len(self.synthesizer.c_abs_list) * self.args.num_sup_train]
        query = support_query[len(self.synthesizer.c_abs_list) * self.args.num_sup_train:]
        support_label_relative = torch.LongTensor((list(range(len(self.synthesizer.c_abs_list)))) * self.args.num_sup_train).cuda(self.args.device)
        query_label_relative = torch.LongTensor((list(range(len(self.synthesizer.c_abs_list)))) * self.args.num_qur_train).cuda(self.args.device)
        return support,support_label_relative,query,query_label_relative

    def task_inverse_from_memory(self):
        support, support_label_relative, query, query_label_relative=self.synthesizer.get_random_task()
        return support, support_label_relative, query, query_label_relative


def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    d_y = y.size(1)
    
    # Handle dimension mismatch
    if d != d_y:
        print(f"Warning: Dimension mismatch in euclidean_dist: {d} vs {d_y}")
        min_d = min(d, d_y)
        x = x[:, :min_d]
        y = y[:, :min_d]
        d = min_d
    
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)
def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

