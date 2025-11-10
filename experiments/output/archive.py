import os
import shutil
from pathlib import Path

def archive_experiment_results(archive_folder_name: str):
    """
    Archive all run_*.json files from epic and story experiments into a named folder.
    
    Parameters:
    -----------
    archive_folder_name : str
        Name of the folder to create for archiving results
    """
    # Define paths
    output_base = Path("experiments/output")
    datasets = ["alfred", "retro", "trident"]
    experiment_types = ["epic", "story"]
    
    # Track statistics
    total_files_moved = 0
    total_folders_created = 0
    
    print(f"\n{'='*60}")
    print(f"Starting archival process: '{archive_folder_name}'")
    print(f"{'='*60}\n")
    
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        
        for exp_type in experiment_types:
            # Source directory with run_*.json files
            source_dir = output_base / dataset / exp_type
            
            if not source_dir.exists():
                print(f"  ⚠️  {exp_type}: Directory not found, skipping...")
                continue
            
            # Get all run_*.json files in the source directory
            run_files = list(source_dir.glob("run_*.json"))
            
            if not run_files:
                print(f"  ℹ️  {exp_type}: No run files found, skipping...")
                continue
            
            # Create archive directory
            archive_dir = source_dir / archive_folder_name
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Move files
            files_moved = 0
            for run_file in run_files:
                destination = archive_dir / run_file.name
                shutil.move(str(run_file), str(destination))
                files_moved += 1
            
            print(f"  ✓ {exp_type}: Moved {files_moved} file(s) to '{archive_folder_name}'")
            total_files_moved += files_moved
            total_folders_created += 1
    
    print(f"\n{'='*60}")
    print(f"Archival complete!")
    print(f"  • Total files moved: {total_files_moved}")
    print(f"  • Archive folders created: {total_folders_created}")
    print(f"{'='*60}\n")

def main():
    """Main function to run the archival script."""
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS ARCHIVAL TOOL")
    print("="*60)
    print("\nThis script will move all 'run_*.json' files from:")
    print("  experiments/output/{dataset}/{epic|story}/")
    print("\nInto a new subfolder with your chosen name.\n")
    
    # Get archive folder name from user
    while True:
        archive_name = input("Enter archive folder name: ").strip()
        
        if not archive_name:
            print("❌ Archive name cannot be empty. Please try again.\n")
            continue
        
        # Check for invalid characters
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        if any(char in archive_name for char in invalid_chars):
            print(f"❌ Archive name contains invalid characters: {invalid_chars}")
            print("Please use a valid folder name.\n")
            continue
        
        break
    
    # Confirm action
    print(f"\n⚠️  You are about to archive results to: '{archive_name}'")
    confirm = input("Continue? (yes/no): ").strip().lower()
    
    if confirm not in ['yes', 'y']:
        print("\n❌ Archival cancelled.")
        return
    
    # Perform archival
    try:
        archive_experiment_results(archive_name)
        print("✅ All operations completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during archival: {e}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()