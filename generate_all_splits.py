"""
Script to generate CSV split files for all datasets on Windows
Run this script to prepare the dataset split files before training
"""
import os
import sys

def run_script(script_path, dataset_name):
    """Run a dataset split generation script"""
    print(f"\n{'='*60}")
    print(f"Generating splits for {dataset_name}...")
    print(f"{'='*60}")
    
    try:
        # Import and run the script
        exec(open(script_path).read(), {'__file__': script_path, '__name__': '__main__'})
        print(f"[OK] {dataset_name} splits generated successfully!")
        return True
    except FileNotFoundError as e:
        print(f"[SKIP] Warning: Data not found for {dataset_name}: {e}")
        print(f"  Skipping {dataset_name}...")
        return False
    except Exception as e:
        print(f"[ERROR] Error generating {dataset_name} splits: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Generate all dataset splits"""
    print("="*60)
    print("Dataset Split Generation Script for Windows")
    print("="*60)
    
    # Define all dataset generation scripts
    scripts = [
        ('write_file/write_flower_filelist.py', 'VGG Flower'),
        ('write_file/write_cub_filelist.py', 'CUB-200-2011'),
        ('write_file/write_eurosat_filelist.py', 'EuroSAT'),
        ('write_file/write_isic_filelist.py', 'ISIC'),
        ('write_file/write_miniimagenet_filelist.py', 'MiniImageNet'),
    ]
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for script_path, dataset_name in scripts:
        if os.path.exists(script_path):
            result = run_script(script_path, dataset_name)
            if result:
                success_count += 1
            elif result is False:
                skipped_count += 1
            else:
                failed_count += 1
        else:
            print(f"[ERROR] Script not found: {script_path}")
            failed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  [OK] Successful: {success_count}")
    print(f"  [SKIP] Skipped: {skipped_count}")
    print(f"  [ERROR] Failed: {failed_count}")
    print(f"{'='*60}\n")
    
    if success_count > 0:
        print("You can now run the main training script!")
        print("\nExample command:")
        print("python main.py --multigpu 0 --gpu 0 --dataset flower --testdataset flower \\")
        print("  --backbone base_clip_16 --method pre_dfmeta_ft --episode_train 240000 \\")
        print("  --pre_datapool_path ./pre_datapool/flower")
    
    return success_count, failed_count, skipped_count

if __name__ == '__main__':
    main()

