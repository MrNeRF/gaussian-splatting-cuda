#!/usr/bin/env python3
import os
import sys
import urllib.request
import urllib.error
import zipfile
import argparse
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Dataset URLs
DATASETS = {
    "mipnerf360": {
        "url": "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
        "folder": "360_v2"
    },
    "mipnerf360_extra": {
        "url": "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip", 
        "folder": "360_extra"
    },
    "bilarf": {
        "url": "https://huggingface.co/datasets/Yuehao/bilarf_data/resolve/main/bilarf_data.zip",
        "folder": "bilarf_data"
    },
    "zipnerf": {
        "urls": [
            "https://storage.googleapis.com/gresearch/refraw360/zipnerf/berlin.zip",
            "https://storage.googleapis.com/gresearch/refraw360/zipnerf/london.zip", 
            "https://storage.googleapis.com/gresearch/refraw360/zipnerf/nyc.zip",
            "https://storage.googleapis.com/gresearch/refraw360/zipnerf/alameda.zip"
        ],
        "folder": "zipnerf"
    },
    "zipnerf_undistorted": {
        "urls": [
            "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/berlin.zip",
            "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/london.zip",
            "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/nyc.zip", 
            "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/alameda.zip"
        ],
        "folder": "zipnerf_undistorted"
    }
}

class DownloadProgress:
    """Thread-safe progress tracking for multiple downloads"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.downloads = {}
        self.last_update = 0
        
    def update_progress(self, filename, downloaded, total_size, speed):
        with self.lock:
            self.downloads[filename] = {
                'downloaded': downloaded,
                'total': total_size,
                'speed': speed,
                'percent': (downloaded / total_size * 100) if total_size > 0 else 0
            }
            
            # Update display every 0.5 seconds to reduce flicker
            current_time = time.time()
            if current_time - self.last_update > 0.5:
                self.display_progress()
                self.last_update = current_time
                
    def display_progress(self):
        """Display clean, stable progress for all downloads"""
        if not self.downloads:
            return
            
        # Clear previous lines
        print(f"\r\033[{len(self.downloads)}A\033[J", end="")
        
        for filename, progress in self.downloads.items():
            bar_width = 30
            filled = int(bar_width * progress['percent'] / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            size_str = f"{self.format_size(progress['downloaded'])}"
            if progress['total'] > 0:
                size_str += f"/{self.format_size(progress['total'])}"
                
            speed_str = f"{self.format_size(progress['speed'])}/s"
            
            print(f"🚀 {filename:15} [{bar}] {progress['percent']:5.1f}% {size_str:>15} {speed_str:>10}")
            
    def finish_download(self, filename, total_downloaded, elapsed_time):
        with self.lock:
            if filename in self.downloads:
                del self.downloads[filename]
                
            avg_speed = total_downloaded / elapsed_time if elapsed_time > 0 else 0
            print(f"✅ {filename:15} Complete! {self.format_size(total_downloaded)} in {self.format_time(elapsed_time)} ({self.format_size(avg_speed)}/s)")
            
    @staticmethod
    def format_size(bytes_size):
        """Format bytes as human readable string"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f}{unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f}TB"
        
    @staticmethod 
    def format_time(seconds):
        """Format seconds as human readable time"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m{seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h{minutes:.0f}m"

# Global progress tracker
progress_tracker = DownloadProgress()

def download_file_fast(url, filepath, max_retries=3):
    """Download a file with resume capability and progress tracking"""
    
    filename = filepath.name
    
    # Check if file already exists and get current size
    current_size = 0
    if filepath.exists():
        current_size = filepath.stat().st_size
        
    start_time = time.time()
    last_update_time = start_time
    
    try:
        req = urllib.request.Request(url)
        if current_size > 0:
            req.add_header('Range', f'bytes={current_size}-')
            
        with urllib.request.urlopen(req) as response:
            total_size = current_size
            if 'Content-Length' in response.headers:
                total_size += int(response.headers['Content-Length'])
            elif 'Content-Range' in response.headers:
                range_info = response.headers['Content-Range']
                total_size = int(range_info.split('/')[-1])
                
            downloaded = current_size
            
            # Open file in appropriate mode
            mode = 'ab' if current_size > 0 else 'wb'
            
            with open(filepath, mode) as f:
                chunk_size = 64 * 1024  # 64KB chunks
                
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                        
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress every 0.5 seconds
                    current_time = time.time()
                    if current_time - last_update_time > 0.5:
                        elapsed = current_time - start_time
                        speed = (downloaded - current_size) / elapsed if elapsed > 0 else 0
                        progress_tracker.update_progress(filename, downloaded, total_size, speed)
                        last_update_time = current_time
                        
            # Final update
            elapsed = time.time() - start_time
            progress_tracker.finish_download(filename, downloaded, elapsed)
            return True
            
    except urllib.error.HTTPError as e:
        if e.code == 416:  # Range not satisfiable - file already complete
            print(f"✅ {filename:15} Already downloaded")
            return True
        else:
            print(f"❌ {filename:15} HTTP Error {e.code}: {e.reason}")
            return False
    except Exception as e:
        print(f"❌ {filename:15} Download failed: {e}")
        return False

def extract_file_fast(filepath, extract_to):
    """Extract zip files with progress indication"""
    filename = filepath.name
    print(f"📦 {filename:15} Extracting...")
    
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ {filename:15} Extracted successfully")
        return True
        
    except Exception as e:
        print(f"❌ {filename:15} Extraction failed: {e}")
        return False

def download_single_file(url, save_path, extract_path):
    """Download and extract a single file"""
    filename = Path(url).name
    filepath = save_path / filename
    
    if not download_file_fast(url, filepath):
        return False
        
    if not extract_file_fast(filepath, extract_path):
        return False
        
    # Clean up downloaded file
    try:
        filepath.unlink()
        print(f"🗑️  {filename:15} Cleaned up")
    except OSError:
        print(f"⚠️  {filename:15} Could not remove archive")
        
    return True

def download_dataset(dataset_name, save_dir):
    """Download and extract a dataset"""
    if dataset_name not in DATASETS:
        print(f"❌ Unknown dataset: {dataset_name}")
        print(f"📋 Available datasets: {', '.join(DATASETS.keys())}")
        return False
    
    dataset_info = DATASETS[dataset_name]
    save_path = Path(save_dir) / dataset_info["folder"]
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Downloading dataset: {dataset_name}")
    print(f"📁 Save location: {save_path}")
    print()
    
    # Handle single URL
    if "url" in dataset_info:
        url = dataset_info["url"]
        return download_single_file(url, save_path, save_path)
        
    # Handle multiple URLs - download sequentially for cleaner progress
    elif "urls" in dataset_info:
        urls = dataset_info["urls"]
        print(f"📦 Files to download: {len(urls)}")
        print()
        
        success_count = 0
        
        for i, url in enumerate(urls, 1):
            print(f"📥 File {i}/{len(urls)}: {Path(url).name}")
            
            if download_single_file(url, save_path, save_path):
                success_count += 1
            else:
                print(f"❌ Failed to download: {Path(url).name}")
            
            print()  # Add spacing between files
        
        if success_count == len(urls):
            print(f"🎉 All {success_count} files downloaded successfully!")
            print(f"📂 Dataset '{dataset_name}' is ready at: {save_path}")
            return True
        else:
            print(f"⚠️  Only {success_count}/{len(urls)} files downloaded successfully")
            return False
    
    return False

def main():
    parser = argparse.ArgumentParser(
        description="🚀 Fast Dataset Downloader for Gaussian Splatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s mipnerf360
  %(prog)s zipnerf --save-dir ./datasets
  %(prog)s --list
        """
    )
    parser.add_argument("dataset", nargs="?", 
                       help=f"Dataset to download. Options: {', '.join(DATASETS.keys())}")
    parser.add_argument("--save-dir", default="./data", 
                       help="Directory to save datasets (default: ./data)")
    parser.add_argument("--list", action="store_true",
                       help="List available datasets")
    
    args = parser.parse_args()
    
    if args.list:
        print("📋 Available datasets:")
        for name, info in DATASETS.items():
            if "url" in info:
                print(f"  🎯 {name}: Single file")
            else:
                print(f"  📦 {name}: {len(info['urls'])} files")
        return 0
    
    if not args.dataset:
        parser.print_help()
        return 1
    
    try:
        if not download_dataset(args.dataset, args.save_dir):
            return 1
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Download interrupted by user")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
