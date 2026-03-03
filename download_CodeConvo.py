"""
Script to examine and download CodeConvo dataset from Hugging Face
Uses huggingface_hub library for inspection and download
"""

from huggingface_hub import (
    list_repo_files,
    hf_hub_download,
    snapshot_download
)
from datasets import load_dataset
import os
import shutil
import traceback
from pathlib import Path
import argparse

def examine_repo_structure():
    """Examine the structure of the dataset repository"""
    print("\n" + "="*70)
    print("EXAMINING REPOSITORY STRUCTURE")
    print("="*70)
    
    repo_id = "jiebi/CodeConvo"
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


def _filter_files_for_request(files, split=None, repo=None, direction=None):
    """Filter repository files based on requested split/repo/direction."""
    if not split:
        return files

    split = split.lower()

    # --split train: download train/{direction}/...
    if split == "train":
        direction = direction or "c2i"
        prefix = f"train/{direction}/"
        return [f for f in files if f.startswith(prefix)]

    # --split dev|test: download {repo}/{direction}/{split}/... and optional flat jsonl files
    if split in ["dev", "test"]:
        prefix = f"{repo}/{direction}/{split}/"
        flat_jsonl_prefix = f"{repo}/{repo}.{direction}.{split}"
        return [
            f for f in files
            if f.startswith(prefix) or f.startswith(flat_jsonl_prefix)
        ]

    return []


def download_dataset(split=None, repo=None, direction=None):
    """Download full dataset or a requested subset from Hugging Face.

    Uses hf_hub_download() with force_download=True to ensure actual file download
    (not symlinks or LFS pointers).
    """
    print("\n" + "="*70)
    print("DOWNLOADING DATASET")
    print("="*70)
    
    repo_id = "jiebi/CodeConvo"
    local_dir = "./dataset/CodeConvo"
    
    if split:
        scope = f"{split}_{repo or 'all'}_{direction or 'default'}"
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
        print(f"Requested split={split}, repo={repo or 'N/A'}, direction={direction or 'N/A'}")
    else:
        print(f"\nDownloading entire dataset to: {local_dir}")
    print("-" * 70)
    
    try:
        # Step 1: List all files in the repository
        print("\nStep 1: Listing all files in repository...")
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        print(f"✓ Found {len(files)} files")
        
        # Filter out certain non-essential files
        skip_patterns = ['.gitattributes', 'README.md', '.huggingface']
        candidate_files = [f for f in files if not any(f.startswith(p) for p in skip_patterns)]
        files_to_download = _filter_files_for_request(
            candidate_files,
            split=split,
            repo=repo,
            direction=direction,
        )

        if not files_to_download:
            print("✗ No files matched the requested selection")
            return False

        print(f"  Will download {len(files_to_download)} files (after filtering)")
        
        # Step 2: Download each file individually with force_download=True
        print("\nStep 2: Downloading files (this may take a while)...")
        downloaded_count = 0
        failed_files = []
        
        for i, file_path in enumerate(files_to_download, 1):
            try:
                # Show progress every 10 files
                if i % 10 == 1 or i == len(files_to_download):
                    print(f"  [{i}/{len(files_to_download)}] Downloading {file_path}...", end=" ", flush=True)
                    show_status = True
                else:
                    show_status = False
                
                # Download file with force_download=True to ensure actual download
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="dataset",
                    cache_dir="./dataset/.cache",
                    force_download=True,  # Force actual download, not symlinks
                    force_filename=None
                )
                
                # Create target directory structure
                target_file = os.path.join(local_dir, file_path)
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                
                # Copy downloaded file to target location
                import shutil
                shutil.copy2(downloaded_path, target_file)
                
                if show_status:
                    file_size = os.path.getsize(target_file)
                    size_str = f"{file_size/1024/1024:.1f}MB" if file_size > 1024*1024 else f"{file_size/1024:.1f}KB"
                    print(f"✓ ({size_str})")
                
                downloaded_count += 1
                
            except Exception as file_error:
                failed_files.append((file_path, str(file_error)[:50]))
                if show_status:
                    print(f"✗")
        
        print(f"\n✓ Downloaded {downloaded_count}/{len(files_to_download)} files")
        
        if failed_files:
            print(f"⚠ {len(failed_files)} files failed to download:")
            for fname, error in failed_files[:5]:
                print(f"   - {fname}: {error}")
            if len(failed_files) > 5:
                print(f"   ... and {len(failed_files) - 5} more")
        
        # Step 3: Clean up cache directory
        print("\nStep 3: Cleaning up cache...")
        cache_dir = "./dataset/.cache"
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print("✓ Cleaned up cache")
        
        # Step 4: Create completion marker
        os.makedirs(local_dir, exist_ok=True)
        with open(completion_marker, 'w') as f:
            f.write("Download completed successfully\n")
        print(f"✓ Created completion marker")
        
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
    
    local_dir = "./dataset/CodeConvo"
    
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


def resolve_data_path(base_dir, split=None, repo=None, direction=None):
    """Resolve a dataset subfolder path based on repo/direction/split.

    Args:
        base_dir: Base dataset directory (e.g., ./dataset/CodeConvo)
        split: Split name (e.g., train, dev, test)
        repo: Repo folder name (e.g., ids, swe, kubernetes) - required for dev/test
        direction: Retrieval direction (e.g., i2c, c2i) - defaults to c2i for train

    Returns:
        Resolved path string or None if invalid combination
        
    Examples:
        - split=train, direction=c2i -> base_dir/train/c2i/
        - split=train, direction=i2c -> base_dir/train/i2c/
        - split=test, repo=kubernetes, direction=i2c -> base_dir/kubernetes/i2c/test
    """
    if not split:
        return base_dir
    
    # For train split, path is train/{direction}
    # Default direction to c2i for train
    if split.lower() == "train":
        direction = direction or "c2i"
        path = os.path.join(base_dir, "train", direction)
        return path
    
    # For dev/test splits, require repo and direction
    # Path structure: {repo}/{direction}/{split}
    if split.lower() in ["dev", "test"]:
        if not repo or not direction:
            return None
        path = os.path.join(base_dir, repo, direction, split)
        return path
    
    # Unknown split
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect CodeConvo structure and download full dataset or a selected subset.",
        epilog="Examples:\n"
               "  # Inspect structure only\n"
               "  python download_CodeConvo.py --no-download\n\n"
               "  # Download entire dataset\n"
               "  python download_CodeConvo.py\n\n"
               "  # Download only train files (defaults to c2i)\n"
               "  python download_CodeConvo.py --split train\n"
               "  python download_CodeConvo.py --split train --direction i2c\n\n"
               "  # Download only dev/test files (requires --repo and --direction)\n"
               "  python download_CodeConvo.py --split test --repo kubernetes --direction i2c\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "test"],
        help="Split name to resolve path for.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        help="Repo folder name (only valid for dev/test splits).",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["i2c", "c2i"],
        help="Retrieval direction. For train: defaults to c2i. For dev/test: required.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading files and only inspect/resolve folder path.",
    )
    return parser.parse_args()


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("CODECONVO DATASET DOWNLOADER")
    print("Repository: https://huggingface.co/datasets/jiebi/CodeConvo")
    print("="*70)

    args = parse_args()
    base_dir = "./dataset/CodeConvo"
    
    # Validate arguments
    if args.split == "train":
        # For train split, only --direction is allowed (no --repo)
        if args.repo:
            print("\n✗ ERROR: --repo is not allowed when --split is 'train'")
            print(f"\nUsage: python download_CodeConvo.py --split train [--direction <i2c|c2i>]")
            print("Note: --direction defaults to 'c2i' if not specified")
            return False
    elif args.split in ["dev", "test"]:
        # For dev/test splits, require both --repo and --direction
        if not args.repo or not args.direction:
            print("\n✗ ERROR: --repo and --direction are required when --split is 'dev' or 'test'")
            print(f"\nUsage: python download_CodeConvo.py --split {args.split} --repo <repo_name> --direction <i2c|c2i>")
            print("\nAvailable repos: ids, ids-supp, swe, kubernetes")
            return False
    
    try:
        # Step 1: Examine repository
        if not examine_repo_structure():
            print("\n⚠ Could not examine repository, but attempting download anyway...")

        # Step 2: Download full dataset or requested subset (unless skipped)
        if not args.no_download:
            if not download_dataset(
                split=args.split,
                repo=args.repo,
                direction=args.direction,
            ):
                print("\n✗ Download failed!")
                return False

        # Step 3: Show downloaded structure
        show_downloaded_structure()

        # Step 4: Resolve and validate specific folder path if split is specified
        if args.split:
            resolved_path = resolve_data_path(
                base_dir, 
                split=args.split, 
                repo=args.repo, 
                direction=args.direction
            )
            
            print("\n" + "="*70)
            print("RESOLVED FOLDER PATH")
            print("="*70)
            print(f"Requested split: {args.split}")
            print(f"Requested repo: {args.repo or 'N/A'}")
            print(f"Requested direction: {args.direction or 'N/A'}")
            print(f"\nResolved path: {resolved_path}")
            
            if resolved_path and os.path.exists(resolved_path):
                print(f"Status: ✓ EXISTS")
                
                # Show contents
                try:
                    items = os.listdir(resolved_path)
                    print(f"\nContents ({len(items)} items):")
                    for item in sorted(items)[:10]:
                        item_path = os.path.join(resolved_path, item)
                        if os.path.isdir(item_path):
                            print(f"  📁 {item}/")
                        else:
                            size = os.path.getsize(item_path)
                            size_str = f"{size/1024/1024:.1f}MB" if size > 1024*1024 else f"{size/1024:.1f}KB"
                            print(f"  📄 {item} ({size_str})")
                    if len(items) > 10:
                        print(f"  ... and {len(items) - 10} more items")
                except Exception as e:
                    print(f"  (Could not list contents: {e})")
            else:
                print(f"Status: ✗ DOES NOT EXIST")
                print(f"\nThe specified path was not found in the downloaded dataset.")
                print(f"Please verify the dataset structure and your arguments.")
        
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
