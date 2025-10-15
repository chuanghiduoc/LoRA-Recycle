"""
Benchmark Table 4: Complexity analysis of inversion
Measures: Accuracy (5w1s, 5w5s), Throughput, FLOPs, GPU Memory
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
import numpy as np
from lora import LoRA_clipModel
from tool import get_dataloader, get_model, setup_seed, data2supportquery
from method.pre_dfmeta_ft import pre_dfmeta_ft
from double_efficient_vit import apply_patch, ReduceEncoder
from tool import find_non_zero_patches
from thop import profile, clever_format


def get_gpu_memory_gb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def reset_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def benchmark_pruning_strategy(args, prune_layer, prune_ratio, num_tasks=100):
    """Benchmark với pruning strategy cụ thể"""
    strategy_name = f"{{{prune_layer}: {prune_ratio}}}"
    print(f"\n{'='*60}")
    print(f"Pruning Strategy: {strategy_name}")
    print(f"{'='*60}")
    
    reset_gpu_memory()
    
    # Update args với pruning config
    args.prune_layer = prune_layer if isinstance(prune_layer, list) else [prune_layer]
    args.prune_ratio = prune_ratio if isinstance(prune_ratio, list) else [prune_ratio]
    
    # Initialize LoRA Recycle với pruning
    lora_recycle = pre_dfmeta_ft(args)
    
    # Test với 5-way 1-shot
    args.way_test = 5
    args.num_sup_test = 1
    args.num_qur_test = 15
    
    accuracies_1shot = []
    start_time = time.time()
    
    for task_id, test_batch in enumerate(lora_recycle.test_loader):
        if task_id >= num_tasks:
            break
        
        data, _ = test_batch[0].to(args.device), test_batch[1].to(args.device)
        support, support_label, query, query_label = data2supportquery(args, data)
        
        accuracy = lora_recycle.test_once(support, support_label, query, query_label)
        accuracies_1shot.append(accuracy)
        
        if (task_id + 1) % 20 == 0:
            print(f"Progress 1-shot: {task_id + 1}/{num_tasks}")
    
    elapsed_1shot = time.time() - start_time
    throughput = num_tasks / elapsed_1shot
    avg_acc_1shot = np.mean(accuracies_1shot)
    
    # Test với 5-way 5-shot
    args.num_sup_test = 5
    accuracies_5shot = []
    
    for task_id, test_batch in enumerate(lora_recycle.test_loader):
        if task_id >= num_tasks:
            break
        
        data, _ = test_batch[0].to(args.device), test_batch[1].to(args.device)
        support, support_label, query, query_label = data2supportquery(args, data)
        
        accuracy = lora_recycle.test_once(support, support_label, query, query_label)
        accuracies_5shot.append(accuracy)
        
        if (task_id + 1) % 20 == 0:
            print(f"Progress 5-shot: {task_id + 1}/{num_tasks}")
    
    avg_acc_5shot = np.mean(accuracies_5shot)
    
    # Measure FLOPs (approximate)
    dummy_input = torch.randn(1, 3, 224, 224).to(args.device)
    with torch.no_grad():
        flops, params = profile(lora_recycle.model.clip_model, inputs=(dummy_input,), verbose=False)
    flops_g = flops / 1e9
    
    # GPU Memory
    gpu_mem = get_gpu_memory_gb()
    
    print(f"5w1s: {avg_acc_1shot:.2f}%, 5w5s: {avg_acc_5shot:.2f}%")
    print(f"Throughput: {throughput:.2f} tasks/s")
    print(f"FLOPs: {flops_g:.2f}G, GPU: {gpu_mem:.2f} GB")
    
    return avg_acc_1shot, avg_acc_5shot, throughput, flops_g, gpu_mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--testdataset', type=str, default='cifar100')
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
    
    import logging
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('timm').setLevel(logging.ERROR)
    
    print("="*80)
    print("BENCHMARK TABLE 4: Complexity analysis of inversion")
    print("Batch size: 25, Dataset: CIFAR-FS")
    print("="*80)
    
    results = {}
    
    # Define pruning strategies
    strategies = [
        ("No Pruning", [0], [0.0]),
        ("{11: 0.75}", [11], [0.75]),
        ("{8: 0.75}", [8], [0.75]),
        ("{6: 0.75}", [6], [0.75]),
        ("Multi-layer", [3, 6, 8, 11], [0.3, 0.3, 0.3, 0.3])
    ]
    
    for name, layers, ratios in strategies:
        acc_1s, acc_5s, thr, flops, mem = benchmark_pruning_strategy(
            args, layers, ratios, num_tasks=100
        )
        results[name] = {
            '5w1s': acc_1s,
            '5w5s': acc_5s,
            'throughput': thr,
            'flops': flops,
            'gpu_mem': mem
        }
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE 4")
    print("="*100)
    print(f"{'Token Pruning Strategy':<30} {'5w 1s':<10} {'5w 5s':<10} {'Throughput':<15} {'FLOPs(G)':<12} {'GPU Mem(GB)':<12}")
    print("-"*100)
    
    baseline = results["No Pruning"]
    
    for name in results:
        r = results[name]
        acc1 = r['5w1s']
        acc5 = r['5w5s']
        thr = r['throughput']
        flops = r['flops']
        mem = r['gpu_mem']
        
        # Calculate improvements
        thr_improve = ((thr - baseline['throughput']) / baseline['throughput'] * 100) if name != "No Pruning" else 0
        flops_improve = ((flops - baseline['flops']) / baseline['flops'] * 100) if name != "No Pruning" else 0
        mem_improve = ((mem - baseline['gpu_mem']) / baseline['gpu_mem'] * 100) if name != "No Pruning" else 0
        
        if name == "No Pruning":
            print(f"{name:<30} {acc1:<10.2f} {acc5:<10.2f} {thr:<15.2f} {flops:<12.2f} {mem:<12.2f}")
        else:
            thr_str = f"{thr:.2f} ({thr_improve:+.0f}%)"
            flops_str = f"{flops:.2f} ({flops_improve:+.0f}%)"
            mem_str = f"{mem:.2f} ({mem_improve:+.0f}%)"
            print(f"{name:<30} {acc1:<10.2f} {acc5:<10.2f} {thr_str:<15} {flops_str:<12} {mem_str:<12}")
    
    print("="*100)
    
    # Save results
    import json
    with open('benchmark_table4_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to: benchmark_table4_results.json")


if __name__ == '__main__':
    main()

