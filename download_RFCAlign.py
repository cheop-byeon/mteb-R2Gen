"""
Script to examine and download RFCAlign dataset from Hugging Face
Uses huggingface_hub library for inspection and download
"""

from huggingface_hub import (
    list_repo_files,
    hf_hub_download,
)
import os
import shutil
import traceback
import argparse

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


def _filter_files_for_request(files, split=None, topic=None):
    """Filter repository files based on requested split/topic."""
    if not split:
        return files

    split_prefix = f"{split}/"
    filtered = [f for f in files if f.startswith(split_prefix)]

    if topic:
        wanted = f"{split}/{topic}.jsonl"
        filtered = [f for f in filtered if f == wanted]

    return filtered


def download_dataset(split=None, topic=None):
    """Download full RFCAlign dataset or a selected subset with real files."""
    print("\n" + "="*70)
    print("DOWNLOADING DATASET")
    print("="*70)
    
    repo_id = "jiebi/RFCAlign"
    local_dir = "./dataset/RFCAlign"
    
    # Scope-aware completion marker
    if split:
        scope = f"{split}_{topic or 'all'}"
        completion_marker = os.path.join(local_dir, f".download_complete_{scope}")
    else:
        completion_marker = os.path.join(local_dir, ".download_complete")

    # Backward compatibility for previous full download marker
    if not split and os.path.exists(completion_marker):
        print("\n✓ Dataset already downloaded (completion marker found)")
        print(f"Location: {local_dir}")
        return True
    
    # Create parent directory
    os.makedirs("./dataset", exist_ok=True)
    
    if split:
        print(f"\nDownloading subset to: {local_dir}")
        print(f"Requested split={split}, topic={topic or 'ALL'}")
    else:
        print(f"\nDownloading full dataset to: {local_dir}")
    print("-" * 70)

    try:
        print("\nStep 1: Listing all files in repository...")
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        print(f"✓ Found {len(files)} files")

        skip_patterns = [".gitattributes", "README.md", ".huggingface"]
        candidate_files = [f for f in files if not any(f.startswith(p) for p in skip_patterns)]
        files_to_download = _filter_files_for_request(candidate_files, split=split, topic=topic)

        if not files_to_download:
            print("✗ No files matched the requested selection")
            return False

        print(f"  Will download {len(files_to_download)} files")

        print("\nStep 2: Downloading files (actual file content, no symlink-only output)...")
        downloaded_count = 0
        failed_files = []

        for i, file_path in enumerate(files_to_download, 1):
            show_status = (i % 10 == 1) or (i == len(files_to_download))
            try:
                if show_status:
                    print(f"  [{i}/{len(files_to_download)}] Downloading {file_path}...", end=" ", flush=True)

                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="dataset",
                    cache_dir="./dataset/.cache",
                    force_download=True,
                    local_dir=local_dir,
                )

                if show_status:
                    file_size = os.path.getsize(downloaded_path)
                    size_str = f"{file_size/1024/1024:.1f}MB" if file_size > 1024*1024 else f"{file_size/1024:.1f}KB"
                    print(f"✓ ({size_str})")

                downloaded_count += 1
            except Exception as file_error:
                failed_files.append((file_path, str(file_error)[:120]))
                if show_status:
                    print("✗")

        print(f"\n✓ Downloaded {downloaded_count}/{len(files_to_download)} files")

        if failed_files:
            print(f"⚠ {len(failed_files)} files failed to download:")
            for fname, err in failed_files[:5]:
                print(f"   - {fname}: {err}")
            if len(failed_files) > 5:
                print(f"   ... and {len(failed_files) - 5} more")

        if downloaded_count == 0:
            print("\n✗ No files were downloaded successfully")
            return False

        print("\nStep 3: Cleaning up cache...")
        cache_dir = "./dataset/.cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("✓ Cleaned up cache")

        local_hf_meta_dir = os.path.join(local_dir, ".cache")
        if os.path.exists(local_hf_meta_dir):
            shutil.rmtree(local_hf_meta_dir)
            print("✓ Cleaned up local HF metadata cache")

        os.makedirs(local_dir, exist_ok=True)
        with open(completion_marker, 'w') as f:
            f.write("Download completed successfully\n")
        print("✓ Created completion marker")

        return downloaded_count > 0

    except Exception as e:
        print(f"\n✗ Download failed: {type(e).__name__}")
        print(f"Message: {str(e)}")
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect RFCAlign structure and download full dataset or a selected subset.",
        epilog="Examples:\n"
               "  # Inspect structure only\n"
               "  python download_RFCAlign.py --no-download\n\n"
               "  # Download everything\n"
               "  python download_RFCAlign.py\n\n"
               "  # Download one split folder only\n"
               "  python download_RFCAlign.py --split qwen_verbose\n\n"
               "  # Download one file only\n"
               "  python download_RFCAlign.py --split qwen_verbose --topic ace\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["llama_non-verbose", "llama_verbose", "qwen_non-verbose", "qwen_verbose"],
        help="Top-level folder to download.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Optional topic file name without .jsonl (requires --split).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading files and only inspect local/remote structure.",
    )
    return parser.parse_args()


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("RFCALIGN DATASET DOWNLOADER")
    print("Repository: https://huggingface.co/datasets/jiebi/RFCAlign")
    print("="*70)
    
    args = parse_args()

    if args.topic and not args.split:
        print("\n✗ ERROR: --topic requires --split")
        return False

    try:
        # Step 1: Examine repository
        if not examine_repo_structure():
            print("\n⚠ Could not examine repository, but attempting download anyway...")
        
        # Step 2: Download dataset (full or filtered)
        if not args.no_download:
            if not download_dataset(split=args.split, topic=args.topic):
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
