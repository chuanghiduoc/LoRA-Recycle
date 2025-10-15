"""
Benchmark script for Table 1: Fine-tuning ViT-B/16 on 5-way 1-shot classification tasks
Measures: Accuracy, Throughput (tasks/s), GPU Memory (GB)
"""

import os
# Tat cac warning va log khong can thiet
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
from transformers import CLIPModel
from lora import LoRA_clipModel
from tool import get_dataloader, get_model, setup_seed, data2supportquery
from method.pre_dfmeta_ft import pre_dfmeta_ft, euclidean_dist
from double_efficient_vit import apply_patch, ReduceEncoder
from tool import find_non_zero_patches


def get_gpu_memory_gb():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def reset_gpu_memory():
    """Reset GPU memory tracking"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def benchmark_full_finetuning(args, optimizer_name, lr, steps, num_tasks=100):
    """Benchmark Full Fine-Tuning"""
    print(f"\n{'='*60}")
    print(f"Full FT - {optimizer_name}, LR={lr}, Steps={steps}")
    print(f"{'='*60}")
    
    reset_gpu_memory()
    _, _, test_loader = get_dataloader(args, resolution=224)
    
    accuracies = []
    start_time = time.time()
    
    for task_id, test_batch in enumerate(test_loader):
        if task_id >= num_tasks:
            break
        
        # Create fresh model with LoRA wrapper for full fine-tuning
        base_model = get_model(args, load=True)
        model = LoRA_clipModel(base_model, r=args.rank, num_classes=args.way_test)
        # Unfreeze ALL parameters for full fine-tuning
        for param in model.parameters():
            param.requires_grad = True
        model = model.to(args.device)
        apply_patch(model=model.clip_model, prune_layer_list=[-1], prune_ratio_list=[0.0], index_matrix=None)
        
        # Get data
        data, _ = test_batch[0].to(args.device), test_batch[1].to(args.device)
        support, support_label, query, query_label = data2supportquery(args, data)
        
        # Setup optimizer - ALL parameters
        if optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Fine-tune
        criterion = nn.CrossEntropyLoss()
        for step in range(steps):
            optimizer.zero_grad()
            
            # Use prototypical network like LoRA
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
        
        if (task_id + 1) % 10 == 0:
            print(f"Progress: {task_id + 1}/{num_tasks}")
        
        del model
        torch.cuda.empty_cache()
    
    elapsed = time.time() - start_time
    throughput = num_tasks / elapsed
    gpu_mem = get_gpu_memory_gb()
    avg_acc = np.mean(accuracies)
    
    print(f"Accuracy: {avg_acc:.2f}%, Throughput: {throughput:.2f} t/s, GPU: {gpu_mem:.2f} GB")
    return avg_acc, throughput, gpu_mem


def benchmark_lora(args, optimizer_name, lr, steps, num_tasks=100):
    """Benchmark LoRA Fine-Tuning"""
    print(f"\n{'='*60}")
    print(f"LoRA - {optimizer_name}, LR={lr}, Steps={steps}")
    print(f"{'='*60}")
    
    reset_gpu_memory()
    _, _, test_loader = get_dataloader(args, resolution=224)
    
    accuracies = []
    start_time = time.time()
    
    for task_id, test_batch in enumerate(test_loader):
        if task_id >= num_tasks:
            break
        
        # Create LoRA model
        base_model = get_model(args, load=True)
        model = LoRA_clipModel(base_model, r=args.rank, num_classes=args.way_test)
        model = model.to(args.device)
        apply_patch(model=model.clip_model, prune_layer_list=[-1], prune_ratio_list=[0.0], index_matrix=None)
        
        # Get data
        data, _ = test_batch[0].to(args.device), test_batch[1].to(args.device)
        support, support_label, query, query_label = data2supportquery(args, data)
        
        # Setup optimizer (only LoRA params)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if optimizer_name == 'SGD':
            optimizer = optim.SGD(trainable_params, lr=lr)
        else:
            optimizer = optim.Adam(trainable_params, lr=lr)
        
        # Fine-tune
        criterion = nn.CrossEntropyLoss()
        for step in range(steps):
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
        
        if (task_id + 1) % 10 == 0:
            print(f"Progress: {task_id + 1}/{num_tasks}")
        
        del model
        torch.cuda.empty_cache()
    
    elapsed = time.time() - start_time
    throughput = num_tasks / elapsed
    gpu_mem = get_gpu_memory_gb()
    avg_acc = np.mean(accuracies)
    
    print(f"Accuracy: {avg_acc:.2f}%, Throughput: {throughput:.2f} t/s, GPU: {gpu_mem:.2f} GB")
    return avg_acc, throughput, gpu_mem


def benchmark_lora_recycle(args, num_tasks=100):
    """Benchmark LoRA Recycle (tuning-free)"""
    print(f"\n{'='*60}")
    print(f"LoRA Recycle (Tuning-Free)")
    print(f"{'='*60}")
    
    reset_gpu_memory()
    
    # Initialize LoRA Recycle
    lora_recycle = pre_dfmeta_ft(args)
    
    # Try to load checkpoint
    checkpoint_path = f'./checkpoints/pre_dfmeta_ft-{args.dataset}-{args.testdataset}-{args.backbone}-inversion/[-1]-[0.0]--1/55-/bestTestModel.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading: {checkpoint_path}")
        lora_recycle.model.load_state_dict(torch.load(checkpoint_path))
    
    accuracies = []
    start_time = time.time()
    
    for task_id, test_batch in enumerate(lora_recycle.test_loader):
        if task_id >= num_tasks:
            break
        
        data, _ = test_batch[0].to(args.device), test_batch[1].to(args.device)
        support, support_label, query, query_label = data2supportquery(args, data)
        
        # No fine-tuning, just evaluate
        accuracy = lora_recycle.test_once(support, support_label, query, query_label)
        accuracies.append(accuracy)
        
        if (task_id + 1) % 10 == 0:
            print(f"Progress: {task_id + 1}/{num_tasks}")
    
    elapsed = time.time() - start_time
    throughput = num_tasks / elapsed
    gpu_mem = get_gpu_memory_gb()
    avg_acc = np.mean(accuracies)
    
    print(f"Accuracy: {avg_acc:.2f}%, Throughput: {throughput:.2f} t/s, GPU: {gpu_mem:.2f} GB")
    return avg_acc, throughput, gpu_mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='miniimagenet')
    parser.add_argument('--testdataset', type=str, default='miniimagenet')
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
    parser.add_argument('--episode_test', type=int, default=100)
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
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.multigpu
    args.device = torch.device(f'cuda:{args.gpu}')
    setup_seed(42)
    
    # Tat log cua transformers va timm
    import logging
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('timm').setLevel(logging.ERROR)
    
    print("="*80)
    print("BENCHMARK TABLE 1: ViT-B/16 on 100 5-way 1-shot tasks")
    print("="*80)
    
    results = {}
    
    # Full Fine-Tuning
    print("\n### FULL FINE-TUNING ###")
    results['full_ft'] = {}
    for opt in ['SGD', 'Adam']:
        results['full_ft'][opt] = {}
        for steps in [50, 5]:
            results['full_ft'][opt][steps] = {}
            for lr in [0.1, 0.01, 0.001]:
                acc, thr, mem = benchmark_full_finetuning(args, opt, lr, steps, 100)
                results['full_ft'][opt][steps][lr] = {'acc': acc, 'thr': thr, 'mem': mem}
    
    # LoRA
    print("\n### LoRA ###")
    results['lora'] = {}
    for opt in ['SGD', 'Adam']:
        results['lora'][opt] = {}
        for steps in [50, 5]:
            results['lora'][opt][steps] = {}
            for lr in [0.1, 0.01, 0.001]:
                acc, thr, mem = benchmark_lora(args, opt, lr, steps, 100)
                results['lora'][opt][steps][lr] = {'acc': acc, 'thr': thr, 'mem': mem}
    
    # LoRA Recycle
    print("\n### LoRA RECYCLE ###")
    acc, thr, mem = benchmark_lora_recycle(args, 100)
    results['lora_recycle'] = {'acc': acc, 'thr': thr, 'mem': mem}
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Method':<20} {'Opt':<8} {'Step':<6} {'0.1':<10} {'0.01':<10} {'0.001':<10} {'Thr(t/s)':<10} {'Mem(GB)':<8}")
    print("-"*80)
    
    for opt in ['SGD', 'Adam']:
        for steps in [50, 5]:
            accs = [results['full_ft'][opt][steps][lr]['acc'] for lr in [0.1, 0.01, 0.001]]
            thr = results['full_ft'][opt][steps][0.01]['thr']
            mem = results['full_ft'][opt][steps][0.01]['mem']
            print(f"{'Full FT':<20} {opt:<8} {steps:<6} {accs[0]:<10.2f} {accs[1]:<10.2f} {accs[2]:<10.2f} {thr:<10.2f} {mem:<8.2f}")
    
    print()
    for opt in ['SGD', 'Adam']:
        for steps in [50, 5]:
            accs = [results['lora'][opt][steps][lr]['acc'] for lr in [0.1, 0.01, 0.001]]
            thr = results['lora'][opt][steps][0.01]['thr']
            mem = results['lora'][opt][steps][0.01]['mem']
            print(f"{'LoRA':<20} {opt:<8} {steps:<6} {accs[0]:<10.2f} {accs[1]:<10.2f} {accs[2]:<10.2f} {thr:<10.2f} {mem:<8.2f}")
    
    print()
    print(f"{'LoRA Recycle':<20} {'—':<8} {'—':<6} {'—':<10} {results['lora_recycle']['acc']:<10.2f} {'—':<10} {results['lora_recycle']['thr']:<10.2f} {results['lora_recycle']['mem']:<8.2f}")
    print("="*80)
    
    # Save results
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark_results.json")


if __name__ == '__main__':
    main()
