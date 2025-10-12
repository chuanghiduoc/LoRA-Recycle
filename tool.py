import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import timm
import time

from dataset.cifar100 import Cifar100
from dataset.samplers import CategoriesSampler
from dataset.miniimagenet import MiniImageNet
from dataset.cub import CUB
from dataset.flower import  flower
from dataset.chest import  chest
from dataset.cropdiseases import  cropdiseases
from dataset.eurosat import  eurosat
from dataset.isic import  isic
from torchvision import transforms
from transformers import  CLIPModel

dataset_classnum={'cifar100':64,'miniimagenet':64,'cub':100,'flower':71,'eurosat':10,'isic':8}
bias={'cifar100':0,'miniimagenet':64,'cub':128,'flower':228,'eurosat':299,'isic':309,'mix':0}
bias_end={'cifar100':63,'miniimagenet':127,'cub':227,'flower':298,'eurosat':308,'isic':316,'mix':316}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dataloader(args,noTransform_test=False,resolution=32,mode='episode'):
    if args.dataset=='cifar100':
        trainset = Cifar100(setname='meta_train', augment=False, resolution=resolution)
        args.num_classes = trainset.num_class
        args.channel = 3
        train_sampler = CategoriesSampler(trainset.label,
                                          args.episode_train,
                                          args.way_train,
                                          args.num_sup_train + args.num_qur_train)
        if mode == 'episode':
            train_loader = DataLoader(dataset=trainset,
                                  num_workers=8,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)
        else:
            train_loader = DataLoader(dataset=trainset,
                                      num_workers=8,
                                      batch_size=(args.way_train*(args.num_sup_train+args.num_qur_train))*args.episode_batch,
                                      shuffle=True,
                                      pin_memory=True)
    elif args.dataset=='miniimagenet':
        trainset = MiniImageNet(setname='meta_train', augment=False, resolution=resolution)
        args.num_classes = trainset.num_class
        args.channel = 3
        train_sampler = CategoriesSampler(trainset.label,
                                          args.episode_train,
                                          args.way_train,
                                          args.num_sup_train + args.num_qur_train)
        if mode == 'episode':
            train_loader = DataLoader(dataset=trainset,
                                  num_workers=8,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)
        else:
            train_loader = DataLoader(dataset=trainset,
                                      num_workers=8,
                                      batch_size=(args.way_train*(args.num_sup_train+args.num_qur_train))*args.episode_batch,
                                      shuffle=True,
                                      pin_memory=True)
    elif args.dataset=='cub':
        trainset = CUB(setname='meta_train', augment=False, resolution=resolution)
        args.num_classes = trainset.num_class
        args.channel = 3
        train_sampler = CategoriesSampler(trainset.label,
                                          args.episode_train,
                                          args.way_train,
                                          args.num_sup_train + args.num_qur_train)
        if mode == 'episode':
            train_loader = DataLoader(dataset=trainset,
                                      num_workers=8,
                                      batch_sampler=train_sampler,
                                      pin_memory=True)
        else:
            train_loader = DataLoader(dataset=trainset,
                                      num_workers=8,
                                      batch_size=(args.way_train * (
                                                  args.num_sup_train + args.num_qur_train)) * args.episode_batch,
                                      shuffle=True,
                                      pin_memory=True)
    elif args.dataset=='flower':
        trainset = flower(setname='meta_train', augment=False, resolution=resolution)
        args.num_classes = trainset.num_class
        args.channel = 3
        train_sampler = CategoriesSampler(trainset.label,
                                          args.episode_train,
                                          args.way_train,
                                          args.num_sup_train + args.num_qur_train)
        if mode == 'episode':
            train_loader = DataLoader(dataset=trainset,
                                      num_workers=8,
                                      batch_sampler=train_sampler,
                                      pin_memory=True)
        else:
            train_loader = DataLoader(dataset=trainset,
                                      num_workers=8,
                                      batch_size=(args.way_train * (
                                                  args.num_sup_train + args.num_qur_train)) * args.episode_batch,
                                      shuffle=True,
                                      pin_memory=True)



    if args.testdataset == 'cifar100':
        valset=Cifar100(setname='meta_val', augment=False, resolution=resolution)
        val_sampler = CategoriesSampler(valset.label,
                                        args.episode_test,
                                          args.way_test,
                                          args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                  num_workers=0,
                                  batch_sampler=val_sampler,
                                  pin_memory=True)
        testset = Cifar100(setname='meta_test', augment=False,noTransform=noTransform_test, resolution=resolution)
        test_sampler = CategoriesSampler(testset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                num_workers=8,
                                batch_sampler=test_sampler,
                                pin_memory=True)
        return train_loader, val_loader, test_loader
    elif args.testdataset == 'miniimagenet':
        valset = MiniImageNet(setname='meta_val', augment=False,resolution=resolution)
        val_sampler = CategoriesSampler(valset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                num_workers=8,
                                batch_sampler=val_sampler,
                                pin_memory=True)
        testset = MiniImageNet(setname='meta_test', augment=False, noTransform=noTransform_test,resolution=resolution)
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return train_loader, val_loader, test_loader
    elif args.testdataset=='cub':
        valset = CUB(setname='meta_val', augment=False,resolution=resolution)
        val_sampler = CategoriesSampler(valset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                num_workers=8,
                                batch_sampler=val_sampler,
                                pin_memory=True)
        testset = CUB(setname='meta_test', augment=False, noTransform=noTransform_test,resolution=resolution)
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return train_loader, val_loader, test_loader
    elif args.testdataset=='flower':
        valset = flower(setname='meta_val', augment=False,resolution=resolution)
        val_sampler = CategoriesSampler(valset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                num_workers=8,
                                batch_sampler=val_sampler,
                                pin_memory=True)
        testset = flower(setname='meta_test', augment=False, noTransform=noTransform_test,resolution=resolution)
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return train_loader, val_loader, test_loader
    elif args.testdataset=='cropdiseases':
        testset = cropdiseases(setname='meta_train', augment=False, noTransform=noTransform_test,resolution=resolution)#only meta_train
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return train_loader, None, test_loader
    elif args.testdataset=='eurosat':
        testset = eurosat(setname='meta_train', augment=False, noTransform=noTransform_test,resolution=resolution)  # only meta_train
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return train_loader, None, test_loader
    elif args.testdataset=='isic':
        testset = isic(setname='meta_train', augment=False, noTransform=noTransform_test,resolution=resolution)  # only meta_train
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return train_loader, None, test_loader
    elif args.testdataset=='chest':
        testset = chest(setname='meta_train', augment=False, noTransform=noTransform_test,resolution=resolution)  # only meta_train
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return train_loader, None, test_loader
    elif args.testdataset=='mix':
        testset_chest = chest(setname='meta_train', augment=False, noTransform=noTransform_test,resolution=resolution)  # only meta_train
        test_sampler_chest = CategoriesSampler(testset_chest.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader_chest = DataLoader(dataset=testset_chest,
                                 num_workers=8,
                                 batch_sampler=test_sampler_chest,
                                 pin_memory=True)

        testset_isic = isic(setname='meta_train', augment=False, noTransform=noTransform_test,
                       resolution=resolution)  # only meta_train
        test_sampler_isic = CategoriesSampler(testset_isic.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader_isic = DataLoader(dataset=testset_isic,
                                 num_workers=8,
                                 batch_sampler=test_sampler_isic,
                                 pin_memory=True)

        testset_eurosat = eurosat(setname='meta_train', augment=False, noTransform=noTransform_test,
                          resolution=resolution)  # only meta_train
        test_sampler_eurosat = CategoriesSampler(testset_eurosat.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader_eurosat = DataLoader(dataset=testset_eurosat,
                                 num_workers=8,
                                 batch_sampler=test_sampler_eurosat,
                                 pin_memory=True)

        testset_cropdiseases = cropdiseases(setname='meta_train', augment=False, noTransform=noTransform_test,
                               resolution=resolution)  # only meta_train
        test_sampler_cropdiseases = CategoriesSampler(testset_cropdiseases.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader_cropdiseases = DataLoader(dataset=testset_cropdiseases,
                                 num_workers=8,
                                 batch_sampler=test_sampler_cropdiseases,
                                 pin_memory=True)
        return None,None,[test_loader_chest,test_loader_isic,test_loader_eurosat,test_loader_cropdiseases]
    else:
        ValueError('not implemented!')
    #return None, val_loader, test_loader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False


def get_model(args,load=True):
    if args.backbone == "base_clip_32":
        if load:
            # Try local path first, fallback to auto-download from HuggingFace
            local_path = os.path.join(os.path.dirname(__file__), "models", "clip-vit-base-patch32")
            if os.path.exists(local_path):
                model = CLIPModel.from_pretrained(local_path, attn_implementation="eager")
            else:
                print("Downloading CLIP ViT-Base-Patch32 from HuggingFace...")
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True, attn_implementation="eager")
        else:
            raise NotImplementedError
    elif args.backbone == "base_clip_16":
        if load:
            # Try local path first, fallback to auto-download from HuggingFace
            local_path = os.path.join(os.path.dirname(__file__), "models", "clip-vit-base-patch16")
            if os.path.exists(local_path):
                model = CLIPModel.from_pretrained(local_path, attn_implementation="eager")
            else:
                print("Downloading CLIP ViT-Base-Patch16 from HuggingFace...")
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True, attn_implementation="eager")
        else:
            raise NotImplementedError
    return model

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def data2supportquery(args,data):
    way = args.way_test
    num_sup = args.num_sup_test
    num_qur = args.num_qur_test
    label = torch.arange(way, dtype=torch.int16).repeat(num_qur+num_sup)
    label = label.type(torch.LongTensor)
    label = label.cuda()
    support=data[:way*num_sup]
    support_label=label[:way*num_sup]
    query=data[way*num_sup:]
    query_label=label[way*num_sup:]
    return support,support_label,query,query_label

def shuffle_task(args,support,support_label,query,query_label):
    support_label_pair=list(zip(support,support_label))
    np.random.shuffle(support_label_pair)
    support,support_label=zip(*support_label_pair)
    support=torch.stack(list(support),dim=0).cuda(args.device)
    support_label=torch.tensor(list(support_label)).cuda(args.device)

    query_label_pair = list(zip(query, query_label))
    np.random.shuffle(query_label_pair)
    query, query_label = zip(*query_label_pair)
    query = torch.stack(list(query), dim=0).cuda(args.device)
    query_label = torch.tensor(list(query_label)).cuda(args.device)

    return support,support_label,query,query_label

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        #x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

NORMALIZE_DICT = {
    'cifar100': dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    'miniimagenet': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'cub': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'mix': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'flower': dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'eurosat':dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'isic':dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'chest':dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'cropdiseases':dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
}
def normalize(tensor, mean, std, reverse=False,keep_zero=True):
    if tensor.dim() not in (3, 4):
        raise ValueError("The input tensor must have 3 or 4 dimensions")

    if keep_zero:
        zero_mask = tensor == 0

    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)

    if tensor.dim() == 4:
        _mean = _mean[None, :, None, None]
        _std = _std[None, :, None, None]
    else:
        _mean = _mean[:, None, None]
        _std = _std[:, None, None]

    tensor = (tensor - _mean) / _std

    if keep_zero:
        tensor[zero_mask] = 0

    return tensor

class Normalizer(object):
    def __init__(self, mean, std,keep_zero=True):
        self.mean = mean
        self.std = std
        self.keep_zero=keep_zero

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse,keep_zero=self.keep_zero)



def label_abs2relative(specific, label_abs):
    trans = dict()
    for relative, abs in enumerate(specific):
        trans[abs] = relative
    label_relative = []
    for abs in label_abs:
        label_relative.append(trans[abs.item()])
    return torch.LongTensor(label_relative)




class CustomColorJitter(transforms.ColorJitter):
    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0.,keep_zero=False):
        super(CustomColorJitter, self).__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.keep_zero=keep_zero

    def __call__(self, img):
        # 应用ColorJitter
        jittered_img = super(CustomColorJitter, self).__call__(img)
        if self.keep_zero:
            # Convert images to numpy arrays for manipulation
            np_img = np.array(img)
            np_jittered_img = np.array(jittered_img)

            # Create a mask where all channels are 0 (black in a 3-channel image)
            black_mask = np.all(np_img == 0, axis=-1)

            # Replace the black pixels in the jittered image with the original black pixels
            np_jittered_img[black_mask] = 0

            # Convert the numpy array back to a PIL image
            jittered_img = Image.fromarray(np_jittered_img)

        return jittered_img

def get_transform(args,dataset=None,aug=False):
    if dataset==None:
        dataset=args.dataset
    if dataset=='cifar100':
        if aug==True:
            transform = transforms.Compose(
                [
                    transforms.RandomChoice([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    CustomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5,keep_zero=True if args.use_mask else False),
                    # transforms.CenterCrop((args.resolution, args.resolution)),
                    ]),
                    transforms.ToTensor(),
                    Normalizer(**NORMALIZE_DICT[args.dataset],keep_zero=True if args.use_mask else False)
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    Normalizer(**NORMALIZE_DICT[args.dataset], keep_zero=True if args.use_mask else False)
                ]
            )
    elif dataset=='miniimagenet':
        if aug==True:
            transform = transforms.Compose(
                [
                    transforms.RandomChoice([
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        CustomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5,keep_zero=True if args.use_mask else False),
                        # transforms.CenterCrop((args.resolution, args.resolution)),
                    ]),
                    # transforms.Resize((args.resolution, args.resolution)),
                    transforms.ToTensor(),
                    Normalizer(**NORMALIZE_DICT[args.dataset],keep_zero=True if args.use_mask else False)
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    # transforms.Resize((args.resolution, args.resolution)),
                    transforms.ToTensor(),
                    Normalizer(**NORMALIZE_DICT[args.dataset],keep_zero=True if args.use_mask else False)
                ]
            )
    elif dataset=='cub':
        if aug == True:
            transform = transforms.Compose(
                [
                    transforms.RandomChoice([
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        CustomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5,keep_zero=True if args.use_mask else False),
                        # transforms.CenterCrop((args.resolution, args.resolution)),
                    ]),
                    # transforms.Resize((args.resolution, args.resolution)),
                    transforms.ToTensor(),
                    Normalizer(**NORMALIZE_DICT[args.dataset], keep_zero=True if args.use_mask else False)
                ]
            )
        else:
            transform = transforms.Compose(
            [
                # transforms.Resize((args.resolution, args.resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    elif dataset=='flower':
        if aug == True:
            transform = transforms.Compose(
                [
                    transforms.RandomChoice([
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        CustomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5,keep_zero=True if args.use_mask else False),
                        # transforms.CenterCrop((args.resolution, args.resolution)),
                    ]),
                    # transforms.Resize((args.resolution, args.resolution)),
                    transforms.ToTensor(),
                    Normalizer(**NORMALIZE_DICT[args.dataset], keep_zero=True if args.use_mask else False)
                ]
            )
        else:
            transform = transforms.Compose(
            [
                transforms.ToTensor(),
                Normalizer(**NORMALIZE_DICT[args.dataset], keep_zero=True if args.use_mask else False)
            ]
        )
    elif dataset=='mix':
        if aug==True:
            transform = transforms.Compose(
                [
                    transforms.RandomChoice([
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        CustomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5,keep_zero=True if args.use_mask else False),
                        # transforms.CenterCrop((args.resolution, args.resolution)),
                    ]),
                    # transforms.Resize((args.resolution, args.resolution)),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    Normalizer(**NORMALIZE_DICT[args.dataset], keep_zero=True if args.use_mask else False)
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    # transforms.Resize((args.resolution, args.resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    Normalizer(**NORMALIZE_DICT[args.dataset], keep_zero=True if args.use_mask else False)
                ]
            )
    else:
        raise NotImplementedError
    return transform







def generate_masked_image_from_index(x, index,attention_scores,patch_size,mask_ratio=0):
    bs, _, _, _ = x.shape
    mask_size = x.shape[2] // patch_size

    if mask_ratio==0:
        one_patch_indices = index
        print(one_patch_indices[0] - 1)
    else:
        bs = attention_scores.shape[0]
        s = attention_scores.shape[1]
        remove_count = int(s * mask_ratio)  # 25%

        _, lowest_attention_indices = torch.topk(attention_scores, remove_count, largest=False, sorted=False, dim=1)

        new_tokens_list = []
        new_indices_list = []

        for i in range(bs):
            mask = torch.ones(s, dtype=torch.bool)
            mask[lowest_attention_indices[i]] = False
            mask = torch.cat((torch.ones(1, dtype=torch.bool), mask), dim=0)

            new_indices = index[i][mask]

            new_indices_list.append(new_indices)

        one_patch_indices = torch.stack(new_indices_list)

        print(one_patch_indices[0], '###')

    mask = torch.zeros(bs, mask_size, mask_size)
    for b in range(bs):
        mask[b, (one_patch_indices[b][1:]-1) // mask_size, (one_patch_indices[b][1:]-1) % mask_size] = 1

    expanded_mask = mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    expanded_mask=expanded_mask.to(x.device)

    masked_image = x * expanded_mask.unsqueeze(1)

    return masked_image

def find_non_zero_patches(images, patch_size):
    bs, c, h, w = images.shape
    patch_h, patch_w = patch_size, patch_size
    if h % patch_h != 0 or w % patch_w != 0:
        raise ValueError("Image dimensions are not divisible by patch size")

    images_reshaped = images.reshape(bs, c, h // patch_h, patch_h, w // patch_w, patch_w)

    images_transposed = images_reshaped.permute(0, 2, 4, 1, 3, 5)

    images_patches = images_transposed.reshape(bs, -1, c * patch_h * patch_w)

    non_zero_patches = torch.any(images_patches != 0, dim=2)

    non_zero_indices = [torch.nonzero(non_zero_patches[i], as_tuple=False).squeeze() + 1 for i in range(bs)]
    non_zero_indices=torch.stack(non_zero_indices,dim=0)
    return non_zero_indices


import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from PIL import Image
from torchvision.datasets import DatasetFolder
from torchvision import transforms


def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)





IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class MemoryFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    def find_classes(self,directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir() and os.listdir(entry))
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        # print(classes)
        return classes, class_to_idx

    def make_dataset(self,
            directory: str,
            class_to_idx: Optional[Dict[str, int]] = None,
            extensions: Optional[Union[str, Tuple[str, ...]]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            allow_empty: bool = False,  # Added for torchvision compatibility
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self.find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes and not allow_empty:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances