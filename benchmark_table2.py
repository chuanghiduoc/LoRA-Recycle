"""
Benchmark Table 2: Recycle in-domain LoRAs
Compare different methods across multiple datasets
FT = fine-tuning-based baselines
FTF = fine-tuning-free baselines
LoRA Recycle_x = using x% token-masked images for meta-training
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import warnings
warnings.filterwarnings('ignore')

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from lora import LoRA_clipModel
from tool import get_dataloader, get_model, setup_seed, data2supportquery, compute_confidence_interval
from method.pre_dfmeta_ft import pre_dfmeta_ft, euclidean_dist
from double_efficient_vit import apply_patch, ReduceEncoder
from tool import find_non_zero_patches


def reset_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def benchmark_full_finetuning(args, num_shot, num_tasks=600):
    """Full Fine-Tuning baseline"""
    print(f"\nFull Fine-Tuning - {num_shot}-shot")
    
    reset_gpu_memory()
    _, _, test_loader = get_dataloader(args, resolution=224)
    
    args.num_sup_test = num_shot
    accuracies = []
    
    for task_id, test_batch in enumerate(test_loader):
        if task_id >= num_tasks:
            break
        
        # Create model
        base_model = get_model(args, load=True)
        model = LoRA_clipModel(base_model, r=args.rank, num_classes=args.way_test)
        for param in model.parameters():
            param.requires_grad = True
        model = model.to(args.device)
        apply_patch(model=model.clip_model, prune_layer_list=[-1], prune_ratio_list=[0.0], index_matrix=None)
        
        # Get data
        data, _ = test_batch[0].to(args.device), test_batch[1].to(args.device)
        support, support_label, query, query_label = data2supportquery(args, data)
        
        # Fine-tune with Adam, lr=0.001, 50 steps (best from Table 1)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for step in range(50):
            optimizer.zero_grad()
            
            ReduceEncoder.init_index_matrix = torch.cat([
                torch.zeros(support.shape[0], 1, dtype=torch.long).to(support.device),
                find_non_zero_patches(images=support, patch_size=model.clip_model.vision_model.embeddings.patch_size)
            ], dim=1)
            z_support = model.get_image_features(support)
            
            ReduceEncoder.init_index_matrix = torch.cat([
                torch.zeros(query.shape[0], 1, dtype=torch.long).to(query.device),
                find_non_zero_patches(images=query, patch_size=model.clip_model.vision_model.embeddings.patch_size)
            ], dim=1)
            z_query = model.get_image_features(query)
            
            z_support = z_support.contiguous().view(args.way_test * args.num_sup_test, -1)
            z_query = z_query.contiguous().view(args.way_test * args.num_qur_test, -1)
            
            protos = []
            for c in range(args.way_test):
                protos.append(z_support[support_label == c].mean(0))
            z_proto = torch.stack(protos, dim=0)
            
            dists = euclidean_dist(z_query, z_proto)
            logits = -dists
            loss = criterion(logits, query_label)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            pat_size = 197
            ReduceEncoder.init_index_matrix = torch.arange(pat_size).repeat(support.shape[0], 1).to(support.device)
            z_support = model.get_image_features(support)
            ReduceEncoder.init_index_matrix = torch.arange(pat_size).repeat(query.shape[0], 1).to(query.device)
            z_query = model.get_image_features(query)
            
            protos = []
            for c in range(args.way_test):
                protos.append(z_support[support_label == c].mean(0))
            z_proto = torch.stack(protos, dim=0)
            
            dists = euclidean_dist(z_query, z_proto)
            logits = -dists
            prediction = torch.max(logits, 1)[1]
            correct = (prediction.cpu() == query_label.cpu()).sum().item()
            accuracy = 100.0 * correct / len(query_label)
            accuracies.append(accuracy)
        
        if (task_id + 1) % 100 == 0:
            print(f"Progress: {task_id + 1}/{num_tasks}")
        
        del model
        torch.cuda.empty_cache()
    
    avg_acc, pm = compute_confidence_interval(accuracies)
    print(f"Result: {avg_acc:.2f}% ± {pm:.2f}%")
    return avg_acc


def benchmark_lora_recycle(args, num_shot, mask_ratio=0, num_tasks=600):
    """LoRA Recycle with optional masking"""
    if mask_ratio == 0:
        print(f"\nLoRA Recycle - {num_shot}-shot (no mask)")
    else:
        print(f"\nLoRA Recycle_{mask_ratio} - {num_shot}-shot ({mask_ratio}% masked)")
    
    reset_gpu_memory()
    
    # Set mask ratio
    args.use_mask = (mask_ratio > 0)
    args.mask_ratio = mask_ratio / 100.0 if mask_ratio > 0 else -1
    
    # Initialize LoRA Recycle
    lora_recycle = pre_dfmeta_ft(args)
    
    # Load checkpoint if available
    checkpoint_path = f'./checkpoints/pre_dfmeta_ft-{args.dataset}-{args.testdataset}-{args.backbone}-inversion/[-1]-[0.0]--1/55-/bestTestModel.pth'
    if os.path.exists(checkpoint_path):
        lora_recycle.model.load_state_dict(torch.load(checkpoint_path))
    
    args.num_sup_test = num_shot
    accuracies = []
    
    for task_id, test_batch in enumerate(lora_recycle.test_loader):
        if task_id >= num_tasks:
            break
        
        data, _ = test_batch[0].to(args.device), test_batch[1].to(args.device)
        support, support_label, query, query_label = data2supportquery(args, data)
        
        accuracy = lora_recycle.test_once(support, support_label, query, query_label)
        accuracies.append(accuracy)
        
        if (task_id + 1) % 100 == 0:
            print(f"Progress: {task_id + 1}/{num_tasks}")
    
    avg_acc, pm = compute_confidence_interval(accuracies)
    print(f"Result: {avg_acc:.2f}% ± {pm:.2f}%")
    return avg_acc


def benchmark_dataset(dataset_name, num_tasks=600):
    """Benchmark all methods on a specific dataset"""
    print(f"\n{'='*80}")
    print(f"BENCHMARKING DATASET: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, default=dataset_name)
    parser.add_argument('--testdataset', type=str, default=dataset_name)
    parser.add_argument('--backbone', type=str, default='base_clip_16')
    parser.add_argument('--resolution', type=int, default=224)
    parser.add_argument('--method', type=str, default='pre_dfmeta_ft')
    parser.add_argument('--episode_batch', type=int, default=1)
    parser.add_argument('--way_train', type=int, default=5)
    parser.add_argument('--num_sup_train', type=int, default=1)
    parser.add_argument('--num_qur_train', type=int, default=15)
    parser.add_argument('--way_test', type=int, default=5)
    parser.add_argument('--num_sup_test', type=int, default=1)
    parser.add_argument('--num_qur_test', type=int, default=15)
    parser.add_argument('--episode_train', type=int, default=5000)
    parser.add_argument('--episode_test', type=int, default=600)
    parser.add_argument('--outer_lr', type=float, default=0.001)
    parser.add_argument('--use_mask', action='store_true', default=False)
    parser.add_argument('--val_interval', type=int, default=2000)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--synthesizer', type=str, default='inversion')
    parser.add_argument('--prune_layer', nargs='+', type=int, default=[-1])
    parser.add_argument('--prune_ratio', nargs='+', type=float, default=[0.0])
    parser.add_argument('--mask_ratio', type=float, default=-1)
    parser.add_argument('--pre_datapool_path', type=str, default=None)
    parser.add_argument('--lora_num', type=int, default=100)
    parser.add_argument('--extra', type=str, default='')
    parser.add_argument('--multigpu', type=str, default='0')
    
    args = parser.parse_args([])
    args.dataset = dataset_name
    args.testdataset = dataset_name
    args.device = torch.device('cuda:0')
    setup_seed(42)
    
    import logging
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('timm').setLevel(logging.ERROR)
    
    results = {}
    
    # FT: Full Fine-Tuning
    print("\n### FT (Fine-Tuning baselines) ###")
    results['Full Finetuning'] = {
        '1shot': benchmark_full_finetuning(args, num_shot=1, num_tasks=num_tasks),
        '5shot': benchmark_full_finetuning(args, num_shot=5, num_tasks=num_tasks)
    }
    
    # FTF: LoRA Recycle variants
    print("\n### FTF (Fine-Tuning-Free baselines) ###")
    
    results['LoRA Recycle'] = {
        '1shot': benchmark_lora_recycle(args, num_shot=1, mask_ratio=0, num_tasks=num_tasks),
        '5shot': benchmark_lora_recycle(args, num_shot=5, mask_ratio=0, num_tasks=num_tasks)
    }
    
    results['LoRA Recycle_25'] = {
        '1shot': benchmark_lora_recycle(args, num_shot=1, mask_ratio=25, num_tasks=num_tasks),
        '5shot': benchmark_lora_recycle(args, num_shot=5, mask_ratio=25, num_tasks=num_tasks)
    }
    
    results['LoRA Recycle_50'] = {
        '1shot': benchmark_lora_recycle(args, num_shot=1, mask_ratio=50, num_tasks=num_tasks),
        '5shot': benchmark_lora_recycle(args, num_shot=5, mask_ratio=50, num_tasks=num_tasks)
    }
    
    results['LoRA Recycle_75'] = {
        '1shot': benchmark_lora_recycle(args, num_shot=1, mask_ratio=75, num_tasks=num_tasks),
        '5shot': benchmark_lora_recycle(args, num_shot=5, mask_ratio=75, num_tasks=num_tasks)
    }
    
    return results


def main():
    print("="*80)
    print("BENCHMARK TABLE 2: Recycle in-domain LoRAs")
    print("Testing on: MiniImageNet, VGG-Flower, CUB")
    print("="*80)
    
    # Datasets to test (excluding CIFAR-FS as requested)
    datasets = ['miniimagenet', 'flower', 'cub']
    all_results = {}
    
    for dataset in datasets:
        results = benchmark_dataset(dataset, num_tasks=600)
        all_results[dataset] = results
    
    # Print summary table
    print("\n" + "="*120)
    print("SUMMARY TABLE 2")
    print("="*120)
    print(f"{'Method':<25} {'MiniImageNet':<25} {'VGG-Flower':<25} {'CUB':<25}")
    print(f"{'':25} {'5-way 1-shot':<12} {'5-way 5-shot':<12} {'5-way 1-shot':<12} {'5-way 5-shot':<12} {'5-way 1-shot':<12} {'5-way 5-shot':<12}")
    print("-"*120)
    
    # FT section
    print("FT")
    method = 'Full Finetuning'
    mini_1s = all_results['miniimagenet'][method]['1shot']
    mini_5s = all_results['miniimagenet'][method]['5shot']
    flower_1s = all_results['flower'][method]['1shot']
    flower_5s = all_results['flower'][method]['5shot']
    cub_1s = all_results['cub'][method]['1shot']
    cub_5s = all_results['cub'][method]['5shot']
    print(f"  {method:<23} {mini_1s:<12.2f} {mini_5s:<12.2f} {flower_1s:<12.2f} {flower_5s:<12.2f} {cub_1s:<12.2f} {cub_5s:<12.2f}")
    
    print()
    
    # FTF section
    print("FTF")
    for method in ['LoRA Recycle', 'LoRA Recycle_25', 'LoRA Recycle_50', 'LoRA Recycle_75']:
        mini_1s = all_results['miniimagenet'][method]['1shot']
        mini_5s = all_results['miniimagenet'][method]['5shot']
        flower_1s = all_results['flower'][method]['1shot']
        flower_5s = all_results['flower'][method]['5shot']
        cub_1s = all_results['cub'][method]['1shot']
        cub_5s = all_results['cub'][method]['5shot']
        print(f"  {method:<23} {mini_1s:<12.2f} {mini_5s:<12.2f} {flower_1s:<12.2f} {flower_5s:<12.2f} {cub_1s:<12.2f} {cub_5s:<12.2f}")
    
    print("="*120)
    
    # Save results
    import json
    with open('benchmark_table2_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to: benchmark_table2_results.json")


if __name__ == '__main__':
    main()

