# This file generates synthetic data from MULTIPLE trained LoRAs (100 LoRAs for paper version)
# NOTE: This is an OPTIONAL step. You can skip this and let meta-training generate on-the-fly.
import math
import os
import shutil
import torch
from torch.utils.data import DataLoader
from dataset.cifar100 import Cifar100_Specific
from dataset.cub import CUB_Specific
from dataset.flower import flower_Specific
from dataset.miniimagenet import MiniImageNet_Specific
from lora import LoRA_clipModel
from synthesis import InversionSyntheiszer
from tool import bias, get_model, get_transform, Normalizer, NORMALIZE_DICT, Timer, dataset_classnum, \
    label_abs2relative, bias_end
from safetensors.torch import load_file


def prepare_model_for_pre_generate(args, load=True):
    """Prepare base model with LoRA structure (for loading trained LoRAs)"""
    model = get_model(args, load=load)
    # NOTE: num_classes should match the way (5 for 5-way)
    model = LoRA_clipModel(model, r=args.rank, num_classes=args.way_test)
    model = model.to(args.device)
    return model

def create_synthesizer(args, teacher, save_dir):
    """Create synthesizer for generating synthetic data"""
    # ULTRA FAST MODE: 100 iterations (for quick generation)
    fast_iterations = 200 if (hasattr(args, 'quick_test') and args.quick_test) else 100  # Changed to 100 for speed
    
    if "32" in args.backbone:
        synthesizer = InversionSyntheiszer(args=args, teacher=teacher,
                                     img_size=(3, args.resolution, args.resolution),
                                     iterations=fast_iterations,
                                     lr_g=0.25,
                                     synthesis_batch_size=None,
                                     adv=0.0, bn=0.01, oh=1.0, tv=0.0, l2=0.0, patch_size=32,
                                     save_dir=save_dir,
                                     transform=get_transform(args, dataset=args.dataset),
                                     normalizer=Normalizer(**NORMALIZE_DICT[args.dataset]),
                                     device=args.device, num_classes=list(range(bias[args.dataset],bias_end[args.dataset]+1)), c_abs_list=None,
                                     max_batch_per_class=10000000)
    elif "16" in args.backbone:
        synthesizer = InversionSyntheiszer(args=args, teacher=teacher,
                                         img_size=(3, args.resolution, args.resolution),
                                         iterations=fast_iterations,
                                         lr_g=0.25,
                                         synthesis_batch_size=None,
                                         adv=0.0, bn=0.01, oh=1.0, tv=0.0, l2=0.0,patch_size=16,
                                         save_dir=save_dir,
                                         transform=get_transform(args, dataset=args.dataset),
                                         normalizer=Normalizer(**NORMALIZE_DICT[args.dataset]),
                                         device=args.device, num_classes=list(range(bias[args.dataset],bias_end[args.dataset]+1)), c_abs_list=None,
                                         max_batch_per_class=10000000)
    return synthesizer

def generate_for_one_lora(args, teacher, synthesizer, lora_id, epoch_id, instance_per_class=10):
    """
    Generate synthetic data for ONE LoRA (used by both standalone and MapReduce)
    
    Args:
        args: Arguments
        teacher: Model with LoRA structure
        synthesizer: InversionSyntheiszer instance
        lora_id: LoRA index
        epoch_id: Epoch index
        instance_per_class: Number of images per class
        
    Returns:
        (success, message, output_dir)
    """
    lora_dir = f'./lorahub/{args.dataset}_{args.backbone}/{args.way_test}way'
    lora_file = f'{lora_dir}/lora_{lora_id}.safetensors'
    label_file = f'{lora_dir}/global_label_{lora_id}.pth'
    
    if not os.path.exists(lora_file):
        return False, f"LoRA file not found: {lora_file}", None
    
    if not os.path.exists(label_file):
        return False, f"Label file not found: {label_file}", None
    
    # Load classes from global_label
    specific = torch.load(label_file, weights_only=False)
    
    print(f"\n[Epoch {epoch_id}][LoRA {lora_id}] Loading {lora_file} for classes {specific}")
    teacher.load_lora_parameters(lora_file)
    
    # Generate synthetic data
    # Convert global labels to relative labels (0, 1, 2, 3, 4 for 5-way)
    relative_labels = list(range(len(specific)))
    synthesizer.c_abs_list = [i + bias[args.dataset] for i in specific]
    support_query_tensor, _ = synthesizer.synthesize(
        targets=torch.LongTensor(relative_labels * instance_per_class),
        student=None, 
        c_num=len(synthesizer.c_abs_list), 
        add=True
    )
    
    return True, f"Successfully generated for LoRA {lora_id}", synthesizer.save_dir

def pre_generate(args):
    """
    Generate synthetic data from multiple trained LoRAs
    
    This creates a pre_datapool that can be used during meta-training to speed up training.
    The meta-training will load 95% from this pool and generate 5% on-the-fly from LoRAs.
    
    Args:
        args.lora_num: Number of LoRAs to use (e.g., 3 for quick test, 100 for paper)
        args.pre_datapool_path: Where to save synthetic images
    """
    timer = Timer()
    
    # Check if LoRAs exist
    lora_dir = f'./lorahub/{args.dataset}_{args.backbone}/{args.way_test}way'
    if not os.path.exists(lora_dir):
        print(f"\n{'='*60}")
        print(f"❌ ERROR: No LoRAs found in {lora_dir}")
        print(f"{'='*60}")
        print("\nYou need to train LoRAs first!")
        print("\nRun this command:")
        print(f"  python train_100_loras.py --dataset {args.dataset} --num_loras {args.lora_num}")
        print("\nOr skip preGenerate and let meta-training generate on-the-fly.")
        return
    
    print(f"\n{'='*60}")
    print(f"Found {args.lora_num} LoRAs in {lora_dir}")
    print(f"Will generate synthetic data from these LoRAs")
    print(f"{'='*60}\n")
    
    # Create output directory
    pre_datapool_path = args.pre_datapool_path
    if pre_datapool_path is None:
        # Set default path if not provided
        pre_datapool_path = f'./pre_datapool/{args.dataset}_{args.backbone}/{args.way_test}way'
        print(f"No pre_datapool_path specified, using default: {pre_datapool_path}\n")
    
    if os.path.exists(pre_datapool_path):
        shutil.rmtree(pre_datapool_path)
        print('Removing old pre_datapool...')
    os.makedirs(pre_datapool_path, exist_ok=True)
    
    # Prepare base model for loading LoRAs
    teacher = prepare_model_for_pre_generate(args, load=True)
    
    # Create synthesizer
    synthesizer = create_synthesizer(args, teacher, pre_datapool_path)

    # NO TRAINING HERE! We use pre-trained LoRAs from train_100_loras.py
    
    #preGenerate from multiple LoRAs
    # Quick test mode: only generate for few classes with few images
    if hasattr(args, 'quick_test') and args.quick_test:
        print("\n" + "="*60)
        print("QUICK TEST MODE ENABLED")
        print(f"Generating {args.test_images_per_class} images for {args.test_num_classes} classes")
        print("="*60 + "\n")
        
        instance_per_class = args.test_images_per_class
        total_classes = min(args.test_num_classes, dataset_classnum[args.dataset])
        epoch = 1  # Only 1 epoch for quick test
        classes_per = total_classes
        lora = 1  # Only 1 batch
    else:
        # Full mode: Generate from ALL 100 LoRAs but with reduced settings
        # FAST MODE: 5 images/class (instead of 10), 1 epoch (instead of 4)
        instance_per_class=5  # Changed from 10 to 5 (2x faster)
        epoch=1  # Changed from 4 to 1 (4x faster)
        classes_per=5
        total_classes = dataset_classnum[args.dataset]
        lora = args.lora_num  # Use ALL LoRAs specified (100)
        
        print(f"\n{'='*60}")
        print(f"ULTRA FAST MODE: Generating from {lora} LoRAs × {epoch} epoch")
        print(f"  - {instance_per_class} images/class")
        print(f"  - 100 iterations (ultra fast!)")
        print(f"Estimated time: ~{lora * epoch * 0.25 / 60:.1f} minutes ({lora * epoch * 0.25 / 3600:.1f} hours)")
        print(f"{'='*60}\n")
    
    for epoch_id in range(epoch):
        for lora_id in range(min(lora, args.lora_num)):  # Don't exceed available LoRAs
            success, message, output_dir = generate_for_one_lora(
                args, teacher, synthesizer, lora_id, epoch_id, instance_per_class
            )
            
            if not success:
                print(f"\n⚠️  WARNING: {message}, skipping...")
                continue
            
            print(f"✅ {message}")
            print('ETA:{}/{}'.format(
                timer.measure(),
                timer.measure(((lora_id+1)*(epoch_id+1)) / (lora*epoch)))
            )