import os

def create_repo_structure():
    # Define the folder hierarchy
    folders = [
        "data/road_surface",
        "data/political_bias",
        "src/models",
        "src/datasets",
        "src/utils",
        "notebooks",
        "outputs/checkpoints",
        "outputs/plots"
    ]
    
    # Files to initialize (with __init__.py for package imports)
    files = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/models/resnet_cnn.py",
        "src/models/vit_transformer.py",
        "src/datasets/__init__.py",
        "src/datasets/road_loader.py",
        "src/datasets/bias_loader.py",
        "src/utils/__init__.py",
        "src/utils/metrics.py",
        "src/utils/plotting.py",
        "main.py",
        "requirements.txt",
        "README.md",
        ".gitignore"
    ]

    # Create directories
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

    # Create empty files
    for file_path in files:
        with open(file_path, 'w') as f:
            if file_path == ".gitignore":
                f.write("data/\noutputs/\n__pycache__/\n*.pth\n.ipynb_checkpoints/")
            elif file_path == "README.md":
                f.write("# CNN vs ViT Comparison\nCollaborators: Cindy Chen & Jasper Matthé")
        print(f"Created file: {file_path}")

if __name__ == "__main__":
    create_repo_structure()
    print("\n✅ Repository structure initialized successfully!")