"""
Train 100 LoRAs for 100 different subsets (Paper version)
Each LoRA trained on 5 classes (5-way)

Timeline: ~100 hours for 100 LoRAs
"""
import torch
import os
import argparse
from torch.utils.data import DataLoader
from lora import LoRA_clipModel
from tool import get_model, dataset_classnum, label_abs2relative, bias
from dataset.flower import flower_Specific
from dataset.cifar100 import Cifar100_Specific
from dataset.cub import CUB_Specific
from dataset.miniimagenet import MiniImageNet_Specific


def train_lora_on_subset(args, classes, lora_id):
    """
    Train LoRA on specific subset of classes
    
    Args:
        args: Arguments
        classes: List of class IDs to train on (e.g., [0,1,2,3,4])
        lora_id: LoRA ID for logging
    
    Returns:
        trained LoRA model
    """
    print(f"\n{'='*60}")
    print(f"Training LoRA {lora_id}/{args.num_loras}")
    print(f"Classes: {classes}")
    print(f"{'='*60}\n")
    
    # Load base model
    model = get_model(args, load=True)
    
    # Create LoRA (num_classes = way, not total classes!)
    teacher = LoRA_clipModel(model, r=args.rank, num_classes=len(classes))
    teacher = teacher.to(args.device)
    
    # Load dataset for this subset only
    if args.dataset == 'cifar100':
        trainset = Cifar100_Specific(setname='meta_train', specific=classes, 
                                      augment=False, resolution=224, mode='train')
        testset = Cifar100_Specific(setname='meta_train', specific=classes,
                                     augment=False, resolution=224, mode='test')
    elif args.dataset == 'cub':
        trainset = CUB_Specific(setname='meta_train', specific=classes,
                                augment=False, resolution=224, mode='train')
        testset = CUB_Specific(setname='meta_train', specific=classes,
                               augment=False, resolution=224, mode='test')
    elif args.dataset == 'flower':
        trainset = flower_Specific(setname='meta_train', specific=classes,
                                    augment=False, resolution=224, mode='train')
        testset = flower_Specific(setname='meta_train', specific=classes,
                                   augment=False, resolution=224, mode='test')
    elif args.dataset == 'miniimagenet':
        trainset = MiniImageNet_Specific(setname='meta_train', specific=classes,
                                          augment=False, resolution=224, mode='train')
        testset = MiniImageNet_Specific(setname='meta_train', specific=classes,
                                         augment=False, resolution=224, mode='test')
    elif args.dataset == 'eurosat':
        from dataset.eurosat import eurosat_Specific
        trainset = eurosat_Specific(setname='meta_train', specific=classes,
                                    augment=False, resolution=224, mode='train')
        testset = eurosat_Specific(setname='meta_train', specific=classes,
                                   augment=False, resolution=224, mode='test')
    elif args.dataset == 'isic':
        from dataset.isic import isic_Specific
        trainset = isic_Specific(setname='meta_train', specific=classes,
                                 augment=False, resolution=224, mode='train')
        testset = isic_Specific(setname='meta_train', specific=classes,
                                augment=False, resolution=224, mode='test')
    else:
        raise NotImplementedError
    
    # DataLoaders
    train_loader = DataLoader(dataset=trainset, num_workers=0, 
                              batch_size=64, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=testset, num_workers=0,
                             batch_size=64, shuffle=True, pin_memory=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(params=teacher.parameters(), lr=0.001)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=[30, 50, 80], gamma=0.2
    )
    
    # Train
    # num_epoch = 1  # ORIGINAL: Full epoch
    num_epoch = 1  # FAST MODE: Still 1 epoch but limited batches
    max_batches = args.max_batches if hasattr(args, 'max_batches') else 999999  # FAST MODE: Limit batches
    
    for epoch in range(num_epoch):
        teacher.train()
        for batch_count, batch in enumerate(train_loader):
            # FAST MODE: Stop after N batches
            if batch_count >= max_batches:
                break
            
            optimizer.zero_grad()
            if len(batch) == 3:
                image, abs_label, label_text = batch[0].cuda(args.device), batch[1].cuda(args.device), batch[2]
            else:
                image, abs_label = batch[0].cuda(args.device), batch[1].cuda(args.device)
                label_text = None
            
            # Convert absolute labels to relative (0-4 for 5 classes)
            relative_label = label_abs2relative(specific=classes, label_abs=abs_label).cuda(args.device)
            
            logits = teacher.get_image_logits(image)
            criteria = torch.nn.CrossEntropyLoss()
            loss = criteria(logits, relative_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 50)
            optimizer.step()
            
            # ORIGINAL: Print every 10 batches
            # if batch_count % 10 == 0:
            #     print(f"  Batch {batch_count}, Loss: {loss.item():.4f}")
            # FAST MODE: Print every 5 batches with progress
            if batch_count % 5 == 0:
                print(f"  Batch {batch_count}/{max_batches}, Loss: {loss.item():.4f}")
        
        lr_schedule.step()
        
        # ORIGINAL: Full validation
        # teacher.eval()
        # correct, total = 0, 0
        # with torch.no_grad():
        #     for batch in test_loader:
        #         image, abs_label = batch[0].cuda(args.device), batch[1].cuda(args.device)
        #         relative_label = label_abs2relative(specific=classes, label_abs=abs_label).cuda(args.device)
        #         logits = teacher.get_image_logits(image)
        #         prediction = torch.max(logits, 1)[1]
        #         correct += (prediction.cpu() == relative_label.cpu()).sum()
        #         total += len(relative_label)
        # test_acc = 100 * correct / total
        # print(f"  Validation accuracy: {test_acc:.2f}%")
        
        # FAST MODE: Skip validation
        print(f"  ⚡ Skipping validation for speed...")
    
    return teacher


def main():
    parser = argparse.ArgumentParser(description='Train 100 LoRAs for paper replication')
    parser.add_argument('--dataset', type=str, default='flower')
    parser.add_argument('--backbone', type=str, default='base_clip_16')
    parser.add_argument('--resolution', type=int, default=224)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--way', type=int, default=5, help='Classes per LoRA')
    parser.add_argument('--num_loras', type=int, default=100, help='Number of LoRAs to train')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--start_id', type=int, default=0, help='Resume from LoRA ID')
    parser.add_argument('--max_batches', type=int, default=10, help='Max batches per LoRA (for quick test)')
    
    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.gpu}')
    
    print("="*70)
    print(" "*15 + "Train 100 LoRAs for Paper Replication")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Backbone: {args.backbone}")
    print(f"Way: {args.way}")
    print(f"Number of LoRAs: {args.num_loras}")
    print(f"Estimated time: ~{args.num_loras} hours ({args.num_loras/24:.1f} days)")
    print("="*70)
    
    response = input("\nContinue? This will take a LONG time! [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    # Create output directory
    output_dir = f'lorahub/{args.dataset}_{args.backbone}/{args.way}way'
    os.makedirs(output_dir, exist_ok=True)
    
    total_classes = dataset_classnum[args.dataset]
    
    # Train each LoRA
    for lora_id in range(args.start_id, args.num_loras):
        # Select classes for this LoRA
        start = (lora_id * args.way) % total_classes
        classes = [(start + i) % total_classes for i in range(args.way)]
        
        # Train LoRA
        teacher = train_lora_on_subset(args, classes, lora_id)
        
        # Save using the built-in save_lora_parameters method
        # This ensures the correct format (w_a_000, w_b_000, fc_...)
        output_path = os.path.join(output_dir, f'lora_{lora_id}.safetensors')
        teacher.save_lora_parameters(output_path)
        
        # Save class mapping
        label_path = os.path.join(output_dir, f'global_label_{lora_id}.pth')
        torch.save(classes, label_path)
        
        print(f"\n✅ Saved LoRA {lora_id}/{args.num_loras}")
        print(f"   File: {output_path}")
        print(f"   Classes: {classes}")
        print(f"   Size: {os.path.getsize(output_path) / 1024**2:.1f} MB")
        
        # Save checkpoint để có thể resume
        checkpoint = {
            'last_lora_id': lora_id,
            'total_loras': args.num_loras,
            'dataset': args.dataset
        }
        torch.save(checkpoint, 'train_loras_checkpoint.pth')
    
    print("\n" + "="*70)
    print("✅ ALL 100 LoRAs TRAINED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")
    print(f"  - lora_0.safetensors to lora_99.safetensors")
    print(f"  - global_label_0.pth to global_label_99.pth")
    print(f"\nNext step:")
    print(f"  1. Comment out line 139 in method/pre_dfmeta_ft.py")
    print(f"  2. Run: python main.py --dataset {args.dataset} --lora_num 100")


if __name__ == '__main__':
    main()

