#!/usr/bin/env python3
"""
Hadoop Mapper for LoRA-Recycle PreGenerate
Each mapper processes one LoRA and generates synthetic images

Input from stdin: lora_id (one per line)
Output to stdout: key-value pairs (lora_id, image_data_path)
"""
import sys
import os
import torch
import json
import base64
import pickle
from io import BytesIO
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.environ.get('LORA_RECYCLE_ROOT', '/home/baotrong/LoRA-Recycle')))

from lora import LoRA_clipModel
from synthesis import InversionSyntheiszer
from tool import get_model, get_transform, Normalizer, NORMALIZE_DICT, bias, bias_end


class MapperArgs:
    """Simulate args object for mapper"""
    def __init__(self, config):
        self.dataset = config['dataset']
        self.backbone = config['backbone']
        self.resolution = config['resolution']
        self.rank = config['rank']
        self.way_test = config['way_test']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.prune_layer = config.get('prune_layer', [-1])
        self.prune_ratio = config.get('prune_ratio', [0.0])
        self.mask_ratio = config.get('mask_ratio', -1)
        self.quick_test = config.get('quick_test', False)
        self.preGenerate = True
        self.lora_num = config.get('lora_num', 100)
        # Additional required attributes
        self.use_mask = config.get('use_mask', False)
        self.multigpu = config.get('multigpu', '0')
        self.gpu = config.get('gpu', 0)
        self.testdataset = config.get('testdataset', self.dataset)
        self.val_interval = config.get('val_interval', 1)
        self.method = config.get('method', 'pre_dfmeta_ft')
        self.episode_batch = config.get('episode_batch', 1)
        self.way_train = config.get('way_train', 5)
        self.num_sup_train = config.get('num_sup_train', 5)
        self.num_qur_train = config.get('num_qur_train', 15)
        self.num_sup_test = config.get('num_sup_test', 5)
        self.num_qur_test = config.get('num_qur_test', 15)
        self.episode_train = config.get('episode_train', 10)
        self.episode_test = config.get('episode_test', 1)
        self.outer_lr = config.get('outer_lr', 0.001)
        self.synthesizer = config.get('synthesizer', 'inversion')
        self.extra = config.get('extra', '')
        self.lorahub = config.get('lorahub', False)
        self.pre_datapool_path = config.get('pre_datapool_path', None)


def prepare_model(args):
    """Prepare base model with LoRA structure"""
    model = get_model(args, load=True)
    model = LoRA_clipModel(model, r=args.rank, num_classes=args.way_test)
    model = model.to(args.device)
    return model


def create_synthesizer(args, teacher, save_dir):
    """Create synthesizer for generating synthetic data"""
    fast_iterations = 200 if (hasattr(args, 'quick_test') and args.quick_test) else 100
    
    if "32" in args.backbone:
        patch_size = 32
    elif "16" in args.backbone:
        patch_size = 16
    else:
        patch_size = 16
    
    synthesizer = InversionSyntheiszer(
        args=args, 
        teacher=teacher,
        img_size=(3, args.resolution, args.resolution),
        iterations=fast_iterations,
        lr_g=0.25,
        synthesis_batch_size=None,
        adv=0.0, bn=0.01, oh=1.0, tv=0.0, l2=0.0,
        patch_size=patch_size,
        save_dir=save_dir,
        transform=get_transform(args, dataset=args.dataset),
        normalizer=Normalizer(**NORMALIZE_DICT[args.dataset]),
        device=args.device,
        num_classes=list(range(bias[args.dataset], bias_end[args.dataset]+1)),
        c_abs_list=None,
        max_batch_per_class=10000000,
        add=False  # Don't auto-save, we'll handle it manually
    )
    return synthesizer


def generate_for_one_lora(args, teacher, synthesizer, lora_id, instance_per_class=5):
    """
    Generate synthetic data for ONE LoRA
    
    Returns:
        dict: {
            'lora_id': int,
            'success': bool,
            'message': str,
            'images_unmasked': list of image tensors,
            'images_masked': list of image tensors,
            'classes': list of class IDs
        }
    """
    # Path to LoRA files - read from local filesystem (shared across cluster nodes)
    # In WSL/Linux, LoRAs are in /mnt/c/LoRA-Recycle/lorahub or similar
    lora_hub_base = os.environ.get('LORA_HUB_DIR', './lorahub')
    
    # Ensure we're using the correct path (WSL vs native Linux)
    if not os.path.exists(lora_hub_base):
        # Try WSL path
        lora_hub_base = '/mnt/c/LoRA-Recycle/lorahub'
        if not os.path.exists(lora_hub_base):
            # Try relative path
            lora_hub_base = os.path.join(os.environ.get('LORA_RECYCLE_ROOT', '.'), 'lorahub')
    
    lora_dir = os.path.join(lora_hub_base, f'{args.dataset}_{args.backbone}/{args.way_test}way')
    lora_file = os.path.join(lora_dir, f'lora_{lora_id}.safetensors')
    label_file = os.path.join(lora_dir, f'global_label_{lora_id}.pth')
    
    if not os.path.exists(lora_file):
        return {
            'lora_id': lora_id,
            'success': False,
            'message': f"LoRA file not found: {lora_file}",
            'images_unmasked': [],
            'images_masked': [],
            'classes': []
        }
    
    if not os.path.exists(label_file):
        return {
            'lora_id': lora_id,
            'success': False,
            'message': f"Label file not found: {label_file}",
            'images_unmasked': [],
            'images_masked': [],
            'classes': []
        }
    
    # Load classes from global_label
    specific = torch.load(label_file, weights_only=False)
    
    # Load LoRA parameters
    teacher.load_lora_parameters(lora_file)
    
    # Generate synthetic data
    relative_labels = list(range(len(specific)))
    synthesizer.c_abs_list = [i + bias[args.dataset] for i in specific]
    
    support_query_tensor_unmasked, support_query_tensor_masked = synthesizer.synthesize(
        targets=torch.LongTensor(relative_labels * instance_per_class),
        student=None,
        c_num=len(synthesizer.c_abs_list),
        add=False  # Don't auto-save
    )
    
    return {
        'lora_id': lora_id,
        'success': True,
        'message': f"Successfully generated for LoRA {lora_id}",
        'images_unmasked': support_query_tensor_unmasked.cpu(),
        'images_masked': support_query_tensor_masked.cpu(),
        'classes': specific,
        'targets': relative_labels * instance_per_class
    }


def serialize_tensor(tensor):
    """Serialize tensor to base64 string"""
    buffer = BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def main():
    """Main mapper function"""
    # Read config from environment variable (set by job submission script)
    config_str = os.environ.get('LORA_MAPREDUCE_CONFIG', '{}')
    config = json.loads(config_str)
    
    if not config:
        print("ERROR: No config found in environment variable LORA_MAPREDUCE_CONFIG", file=sys.stderr)
        sys.exit(1)
    
    # Create args
    args = MapperArgs(config)
    
    # Prepare model (this is heavy, ~500MB)
    print(f"[Mapper] Loading base model on device: {args.device}", file=sys.stderr)
    teacher = prepare_model(args)
    
    # Create synthesizer (without data_pool to avoid file I/O conflicts)
    # We'll save manually after generation
    temp_dir = os.path.join('/tmp', f'mapper_{os.getpid()}')
    os.makedirs(temp_dir, exist_ok=True)
    synthesizer = create_synthesizer(args, teacher, temp_dir)
    
    instance_per_class = config.get('instance_per_class', 5)
    
    # Read lora_ids from stdin
    for line in sys.stdin:
        lora_id = int(line.strip())
        
        print(f"[Mapper] Processing LoRA {lora_id}", file=sys.stderr)
        
        # Generate synthetic data
        result = generate_for_one_lora(args, teacher, synthesizer, lora_id, instance_per_class)
        
        if not result['success']:
            print(f"[Mapper] ERROR: {result['message']}", file=sys.stderr)
            continue
        
        # Save images to temporary local files
        output_dir = os.path.join(temp_dir, f'lora_{lora_id}')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'unmasked'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masked'), exist_ok=True)
        
        # Save unmasked images
        unmasked_tensor = result['images_unmasked']
        masked_tensor = result['images_masked']
        targets = result['targets']
        classes = result['classes']
        
        # Group images by class and save
        from torchvision.utils import save_image
        for class_id in classes:
            class_dir_unmasked = os.path.join(output_dir, 'unmasked', str(class_id + bias[args.dataset]))
            class_dir_masked = os.path.join(output_dir, 'masked', str(class_id + bias[args.dataset]))
            os.makedirs(class_dir_unmasked, exist_ok=True)
            os.makedirs(class_dir_masked, exist_ok=True)
            
            # Get images for this class
            class_indices = [i for i, t in enumerate(targets) if t == class_id]
            
            for idx, img_idx in enumerate(class_indices):
                # Save unmasked
                save_image(unmasked_tensor[img_idx], 
                          os.path.join(class_dir_unmasked, f'{lora_id}_{idx}.png'))
                # Save masked
                save_image(masked_tensor[img_idx],
                          os.path.join(class_dir_masked, f'{lora_id}_{idx}.png'))
        
        # Emit key-value pair: (lora_id, output_dir)
        # Output format: tab-separated key\tvalue
        print(f"{lora_id}\t{output_dir}")
        
        print(f"[Mapper] ✅ Completed LoRA {lora_id}, saved to {output_dir}", file=sys.stderr)


if __name__ == '__main__':
    main()

