from pathlib import Path

def print_dir_tree(directory, prefix=""):
    """Recursively prints a visual tree of directories only."""
    path = Path(directory)
    
    try:
        # Filter for directories only and sort them
        dirs = sorted([item for item in path.iterdir() if item.is_dir()])
    except PermissionError:
        return

    for index, item in enumerate(dirs):
        is_last = (index == len(dirs) - 1)
        connector = "└── " if is_last else "├── "
        
        print(f"{prefix}{connector}{item.name}/")
        
        # Recurse into the subfolder
        next_prefix = prefix + ("    " if is_last else "│   ")
        print_dir_tree(item, next_prefix)

# Target just the "data" folder
target_folder = Path("data")

if target_folder.exists() and target_folder.is_dir():
    print(f"[{target_folder.name}/]")
    print_dir_tree(target_folder)
else:
    print("Error: The folder 'data' does not exist in the current directory.")
