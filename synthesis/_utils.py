import torch
from torch.utils.data import DataLoader
import numpy as np 
from PIL import Image
import os, math
from contextlib import contextmanager
import sys

from dataset.samplers import CategoriesSampler

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from tool import bias, bias_end, MemoryFolder, data2supportquery, shuffle_task

def clip_images(image_tensor, mean, std,use_fp16=False):
    if use_fp16:
        mean = np.array(mean, dtype=np.float16)
        std = np.array(std, dtype=np.float16)
    else:
        mean = np.array(mean)
        std = np.array(std)
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor
    
def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy()*255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir!='':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images( imgs, col=col ).transpose( 1, 2, 0 ).squeeze()
        imgs = Image.fromarray( imgs )
        if size is not None:
            if isinstance(size, (list,tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max( h, w )
                scale = float(size) / float(max_side)
                _w, _h = int(w*scale), int(h*scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output[:-4]
        for idx, img in enumerate(imgs):
            img = Image.fromarray( img.transpose(1, 2, 0).squeeze() )
            img.save(output_filename+'-%d.png'%(idx))

def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple) ):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0,3,1,2) # make it channel first
    assert len(images.shape)==4
    assert isinstance(images, np.ndarray)

    N,C,H,W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))
    
    pack = np.zeros( (C, H*row+padding*(row-1), W*col+padding*(col-1)), dtype=images.dtype )
    for idx, img in enumerate(images):
        h = (idx // col) * (H+padding)
        w = (idx % col) * (W+padding)
        pack[:, h:h+H, w:w+W] = img
    return pack

def flatten_dict(dic):
    flattned = dict()
    def _flatten(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                if prefix is None:
                    _flatten( k, v )
                else:
                    _flatten( prefix+'/%s'%k, v )
            else:
                if prefix is None:
                    flattned[k] = v
                else:
                    flattned[ prefix+'/%s'%k ] = v
        
    _flatten(None, dic)
    return flattned

def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [ -m / s for m, s in zip(mean, std) ]
        _std = [ 1/s for s in std ]
    else:
        _mean = mean
        _std = std
    
    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor

class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)


class ImagePool2(object):
    def __init__(self,args, root,num_classes,transform,max_batch_per_class,use_mask):
        self.args=args
        self.root_all_prune=os.path.join(root,'unmaksed') #for unmasked version
        self.root_all_prune_mask = os.path.join(root, 'masked')#for masked version

        self._idx=dict()
        for c_abs in num_classes:
            self._idx[c_abs]=0
            os.makedirs(os.path.join(self.root_all_prune, str(c_abs)), exist_ok=True)
            os.makedirs(os.path.join(self.root_all_prune_mask, str(c_abs)), exist_ok=True)


        self.max_batch_per_class=max_batch_per_class
        self.ready_class={name:[] for name,start in bias.items()}
        self.transform=transform
        #use pre-inverted image
        if self.args.preGenerate==False:
            # Use masked or unmaksed folder based on use_mask setting
            memory_root = os.path.join(self.args.pre_datapool_path, 'masked' if use_mask else 'unmaksed')
            self.memoryyDataset = MemoryFolder(root=memory_root, transform=self.transform)
            self.train_sampler = CategoriesSampler(self.memoryyDataset.targets,
                                              args.episode_train,
                                              args.way_train,
                                              args.num_sup_train + args.num_qur_train)
            self.memoryLoader = DataLoader(dataset=self.memoryyDataset,
                                          num_workers=8,
                                          batch_sampler=self.train_sampler,
                                          pin_memory=True)
            self.iter=iter(self.memoryLoader)

    def add(self, imgs,mask=False, c_abs_list=None,synthesis_batch_size_per_class=None,c_abs_targets=None):
        if c_abs_targets==None:
            c_abs_targets=torch.LongTensor(c_abs_list*synthesis_batch_size_per_class)
        else:
            pass
        for c_abs in c_abs_list:
            if mask==False:
                root = os.path.join(self.root_all_prune, str(c_abs))
                self._idx[c_abs] += 1
                self._idx[c_abs] = self._idx[c_abs] % self.max_batch_per_class
                for (dataset_name, end_id) in bias_end.items():
                    if c_abs <= end_id:
                        if c_abs not in self.ready_class[dataset_name]:
                            self.ready_class[dataset_name].append(c_abs)
                        break
            else:
                root = os.path.join(self.root_all_prune_mask, str(c_abs))

            imgs_c=imgs[c_abs_targets==c_abs]
            save_image_batch(imgs_c, os.path.join( root, "%d.png"%(self._idx[c_abs]) ), pack=False)


    def get_random_task(self):
        train_batch=next(self.iter)
        data, _ = train_batch[0].cuda(self.args.device), train_batch[1].cuda(self.args.device)
        support, support_label_relative, query, query_label_relative = data2supportquery(self.args, data)
        support, support_label_relative, query, query_label_relative = shuffle_task(self.args, support,support_label_relative, query,query_label_relative)
        return support, support_label_relative, query, query_label_relative



class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)
    
    def next(self):
        try:
            data = next( self._iter )
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next( self._iter )
        return data

@contextmanager
def dummy_ctx(*args, **kwds):
    try:
        yield None
    finally:
        pass