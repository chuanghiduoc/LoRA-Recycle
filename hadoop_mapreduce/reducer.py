#!/usr/bin/env python3
"""
Hadoop Reducer for LoRA-Recycle PreGenerate
Collects all generated images from mappers and merges them into final pre_datapool

Input from stdin: key-value pairs (lora_id, output_dir)
Output: Merged directory structure in HDFS
"""
import sys
import os
import shutil
import json
from collections import defaultdict


def main():
    """Main reducer function"""
    # Read config from environment variable
    config_str = os.environ.get('LORA_MAPREDUCE_CONFIG', '{}')
    config = json.loads(config_str)
    
    if not config:
        print("ERROR: No config found in environment variable LORA_MAPREDUCE_CONFIG", file=sys.stderr)
        sys.exit(1)
    
    dataset = config['dataset']
    
    # Output directory (will be written to HDFS)
    # No need for subfolder structure - meta_train copies the whole directory
    output_root = config.get('output_dir', f'./pre_datapool_mapreduce/{dataset}')
    
    # Create output directories
    os.makedirs(os.path.join(output_root, 'unmasked'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'masked'), exist_ok=True)
    
    print(f"[Reducer] Output directory: {output_root}", file=sys.stderr)
    
    # Collect all mapper outputs
    mapper_outputs = defaultdict(list)
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            lora_id, output_dir = line.split('\t')
            lora_id = int(lora_id)
            mapper_outputs[lora_id].append(output_dir)
        except ValueError as e:
            print(f"[Reducer] ERROR parsing line: {line}, error: {e}", file=sys.stderr)
            continue
    
    print(f"[Reducer] Received outputs from {len(mapper_outputs)} LoRAs", file=sys.stderr)
    
    # Merge all images into output directory
    total_files_copied = 0
    
    for lora_id in sorted(mapper_outputs.keys()):
        output_dirs = mapper_outputs[lora_id]
        
        for output_dir in output_dirs:
            if not os.path.exists(output_dir):
                print(f"[Reducer] WARNING: Directory not found: {output_dir}", file=sys.stderr)
                continue
            
            print(f"[Reducer] Merging LoRA {lora_id} from {output_dir}", file=sys.stderr)
            
            # Copy unmasked images
            unmasked_src = os.path.join(output_dir, 'unmasked')
            if os.path.exists(unmasked_src):
                for class_dir in os.listdir(unmasked_src):
                    src_class_path = os.path.join(unmasked_src, class_dir)
                    dst_class_path = os.path.join(output_root, 'unmasked', class_dir)
                    os.makedirs(dst_class_path, exist_ok=True)
                    
                    # Copy all images
                    for img_file in os.listdir(src_class_path):
                        src_file = os.path.join(src_class_path, img_file)
                        dst_file = os.path.join(dst_class_path, img_file)
                        shutil.copy2(src_file, dst_file)
                        total_files_copied += 1
            
            # Copy masked images
            masked_src = os.path.join(output_dir, 'masked')
            if os.path.exists(masked_src):
                for class_dir in os.listdir(masked_src):
                    src_class_path = os.path.join(masked_src, class_dir)
                    dst_class_path = os.path.join(output_root, 'masked', class_dir)
                    os.makedirs(dst_class_path, exist_ok=True)
                    
                    # Copy all images
                    for img_file in os.listdir(src_class_path):
                        src_file = os.path.join(src_class_path, img_file)
                        dst_file = os.path.join(dst_class_path, img_file)
                        shutil.copy2(src_file, dst_file)
                        total_files_copied += 1
            
            # Clean up temporary directory
            try:
                shutil.rmtree(output_dir)
                print(f"[Reducer] Cleaned up temporary directory: {output_dir}", file=sys.stderr)
            except Exception as e:
                print(f"[Reducer] WARNING: Could not clean up {output_dir}: {e}", file=sys.stderr)
    
    print(f"[Reducer] ✅ Merge complete! Copied {total_files_copied} files to {output_root}", file=sys.stderr)
    
    # Create summary file
    summary = {
        'total_loras_processed': len(mapper_outputs),
        'total_files_copied': total_files_copied,
        'output_directory': output_root,
        'config': config
    }
    
    summary_file = os.path.join(output_root, 'generation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[Reducer] Summary written to {summary_file}", file=sys.stderr)
    
    # Output final summary (this will be visible in job output)
    print(f"SUCCESS\t{json.dumps(summary)}")


if __name__ == '__main__':
    main()

