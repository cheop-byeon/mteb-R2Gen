"""
Script to examine and download RFCAlign dataset from Hugging Face
Uses huggingface_hub library for inspection and download
"""

from huggingface_hub import (
    list_repo_files,
    hf_hub_download,
    snapshot_download
)
from datasets import load_dataset
import os
import traceback
from pathlib import Path

def examine_repo_structure():
    """Examine the structure of the dataset repository"""
    print("\n" + "="*70)
    print("EXAMINING REPOSITORY STRUCTURE")
    print("="*70)
    
    repo_id = "jiebi/RFCAlign"
    repo_type = "dataset"
    
    try:
        # Get basic repo info
        print("\n1. Repository Information:")
        print("-" * 70)
        print(f"Repo ID: {repo_id}")
        print(f"Repo Type: {repo_type}")
        
        # List all files and folders
        print("\n2. Files and Folders in Repository:")
        print("-" * 70)
        files = list_repo_files(repo_id=repo_id, repo_type=repo_type)
        print(f"Total items: {len(files)}\n")
        
        # Organize by folder
        folders = {}
        for file_path in sorted(files):
            if "/" in file_path:
                folder = file_path.split("/")[0]
                if folder not in folders:
                    folders[folder] = []
                folders[folder].append(file_path)
            else:
                if "root" not in folders:
                    folders["root"] = []
                folders["root"].append(file_path)
        
        # Display structure
        for folder in sorted(folders.keys()):
            print(f"\n📁 {folder}/")
            for file_path in sorted(folders[folder])[:10]:  # Show first 10 files
                size_marker = ""
                if file_path.endswith((".parquet", ".jsonl", ".json", ".arrow")):
                    size_marker = " [data file]"
                print(f"   └─ {file_path}{size_marker}")
            if len(folders[folder]) > 10:
                print(f"   └─ ... and {len(folders[folder]) - 10} more files")
        
        print("\n" + "="*70)
        return True
        
    except Exception as e:
        print(f"\n✗ Error examining repository: {type(e).__name__}")
        print(f"Message: {str(e)}")
        traceback.print_exc()
        return False


def download_dataset():
    """Download the dataset using appropriate Hugging Face method"""
    print("\n" + "="*70)
    print("DOWNLOADING DATASET")
    print("="*70)
    
    repo_id = "jiebi/RFCAlign"
    local_dir = "./dataset/RFCAlign"
    
    # Check for completion marker
    completion_marker = os.path.join(local_dir, ".download_complete")
    if os.path.exists(completion_marker):
        print("\n✓ Dataset already downloaded (completion marker found)")
        print(f"Location: {local_dir}")
        return True
    
    # Create parent directory
    os.makedirs("./dataset", exist_ok=True)
    
    print(f"\nDownloading to: {local_dir}")
    print("-" * 70)
    
    try:
        # Method 1: Try snapshot_download with allow_patterns for better control
        print("\n1. Attempting to download with snapshot_download()...")
        print("   (This preserves the folder structure)")
        
        # Download to temporary cache location
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir="./dataset/.cache",
            allow_patterns="*",  # Download all files
        )
        
        print(f"✓ Download successful!")
        print(f"Cache location: {path}")
        
        # Move files from cache to target directory (flatten structure)
        print(f"\nMoving files to {local_dir}...")
        import shutil
        
        # Remove target if it exists
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        
        # Move the downloaded folder to target location
        shutil.move(path, local_dir)
        print(f"✓ Moved to: {local_dir}")
        
        # Clean up cache directory
        cache_dir = "./dataset/.cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"✓ Cleaned up cache")
        
        # Create completion marker
        os.makedirs(local_dir, exist_ok=True)
        with open(completion_marker, 'w') as f:
            f.write("Download completed successfully\n")
        print(f"✓ Created completion marker")
        
        return True
        
    except Exception as e:
        print(f"\n✗ snapshot_download failed: {type(e).__name__}")
        print(f"Message: {str(e)[:200]}")
        
        # Try alternative method
        print("\n2. Attempting alternative: loading individual splits...")
        print("-" * 70)
        
        try:
            # Try loading splits individually
            splits = ["train", "validation", "test"]
            downloaded = []
            
            for split in splits:
                try:
                    print(f"\n   Loading split: {split}...", end=" ")
                    data = load_dataset(repo_id, split=split, trust_remote_code=True)
                    split_path = os.path.join(local_dir, split)
                    data.save_to_disk(split_path)
                    print(f"✓ ({len(data)} samples)")
                    downloaded.append(split)
                except Exception as split_error:
                    print(f"✗ ({str(split_error)[:50]})")
            
            if downloaded:
                print(f"\n✓ Successfully downloaded splits: {', '.join(downloaded)}")
                
                # Create completion marker
                os.makedirs(local_dir, exist_ok=True)
                with open(completion_marker, 'w') as f:
                    f.write("Download completed successfully\n")
                print(f"✓ Created completion marker")
                
                return True
            else:
                print("\n✗ No splits could be downloaded")
                return False
                
        except Exception as alt_error:
            print(f"\n✗ Alternative method also failed: {type(alt_error).__name__}")
            print(f"Message: {str(alt_error)[:200]}")
            traceback.print_exc()
            return False


def show_downloaded_structure():
    """Display the structure of downloaded dataset"""
    print("\n" + "="*70)
    print("DOWNLOADED DATASET STRUCTURE")
    print("="*70)
    
    local_dir = "./dataset/RFCAlign"
    
    if not os.path.exists(local_dir):
        print(f"\n✗ Directory not found: {local_dir}")
        return
    
    print(f"\nLocation: {local_dir}\n")
    
    def show_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        try:
            items = sorted(os.listdir(path))
            dirs = [i for i in items if os.path.isdir(os.path.join(path, i))]
            files = [i for i in items if os.path.isfile(os.path.join(path, i))]
            
            # Show directories
            for i, dir_name in enumerate(dirs[:10]):
                is_last = (i == len(dirs) - 1) and len(files) == 0
                print(f"{prefix}{'└── ' if is_last else '├── '}{dir_name}/")
                
                new_prefix = prefix + ("    " if is_last else "│   ")
                show_tree(os.path.join(path, dir_name), new_prefix, max_depth, current_depth + 1)
            
            if len(dirs) > 10:
                print(f"{prefix}├── ... and {len(dirs) - 10} more directories")
            
            # Show files
            for i, file_name in enumerate(files[:10]):
                is_last = i == len(files) - 1
                size = os.path.getsize(os.path.join(path, file_name))
                size_str = f"({size/1024/1024:.1f}MB)" if size > 1024*1024 else f"({size/1024:.1f}KB)" if size > 1024 else f"({size}B)"
                print(f"{prefix}{'└── ' if is_last else '├── '}{file_name} {size_str}")
            
            if len(files) > 10:
                print(f"{prefix}└── ... and {len(files) - 10} more files")
                
        except PermissionError:
            print(f"{prefix}[Permission Denied]")
    
    show_tree(local_dir)
    print("\n" + "="*70)


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("RFCALIGN DATASET DOWNLOADER")
    print("Repository: https://huggingface.co/datasets/jiebi/RFCAlign")
    print("="*70)
    
    try:
        # Step 1: Examine repository
        if not examine_repo_structure():
            print("\n⚠ Could not examine repository, but attempting download anyway...")
        
        # Step 2: Download dataset
        if not download_dataset():
            print("\n✗ Download failed!")
            return False
        
        # Step 3: Show downloaded structure
        show_downloaded_structure()
        
        print("\n" + "="*70)
        print("✓ COMPLETED SUCCESSFULLY")
        print("="*70)
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("✗ UNEXPECTED ERROR")
        print("="*70)
        print(f"\nException: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
