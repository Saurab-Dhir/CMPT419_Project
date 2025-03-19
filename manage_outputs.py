import os
import shutil
import argparse
import glob
from datetime import datetime

def clear_outputs():
    """Clear all output directories except for the directories themselves."""
    output_dirs = [
        "logs/transcriptions",
        "logs/responses",
        "output/audio",
        "static/audio"
    ]
    
    for directory in output_dirs:
        if os.path.exists(directory):
            print(f"Clearing {directory}...")
            # Delete all files in the directory
            for file_path in glob.glob(os.path.join(directory, "*")):
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"  Deleted: {file_path}")
    
    print("\nAll output directories cleared.")

def create_dirs():
    """Create necessary directories if they don't exist."""
    dirs = [
        "logs",
        "logs/transcriptions",
        "logs/responses",
        "output",
        "output/audio",
        "static",
        "static/audio"
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory ensured: {directory}")
    
    print("\nAll required directories are ready.")

def list_outputs():
    """List all output files across directories."""
    output_dirs = [
        "logs/transcriptions",
        "logs/responses",
        "output/audio",
        "static/audio"
    ]
    
    total_files = 0
    
    for directory in output_dirs:
        if os.path.exists(directory):
            files = glob.glob(os.path.join(directory, "*"))
            file_count = len(files)
            total_files += file_count
            
            print(f"\n{directory} ({file_count} files):")
            if file_count > 0:
                for idx, file_path in enumerate(sorted(files), 1):
                    file_size = os.path.getsize(file_path) / 1024  # Size in KB
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    mod_time_str = mod_time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"  {idx}. {os.path.basename(file_path)} ({file_size:.2f} KB) - {mod_time_str}")
    
    print(f"\nTotal files: {total_files}")

def main():
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(description="Manage output files for the Empathetic Self-talk Coach")
    parser.add_argument("action", choices=["clear", "create", "list"], 
                        help="Action to perform: clear outputs, create directories, or list outputs")
    
    args = parser.parse_args()
    
    if args.action == "clear":
        clear_outputs()
    elif args.action == "create":
        create_dirs()
    elif args.action == "list":
        list_outputs()

if __name__ == "__main__":
    main() 