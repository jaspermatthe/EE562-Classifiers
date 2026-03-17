from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import shutil
import random

def get_road_dl(data_dir, batch_size=128, num_workers=12):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize according to mean and std of ImageNet
        # https://en.wikipedia.org/wiki/ImageNet#:~:text=For%20example%2C%20in%20PyTorch,data.%5B27%5D
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)

def sample_road_surface_dataset(base_path, output_path, train_per_class=160, val_per_class=3, test_per_class=10):
    """
    Sample a subset of the road surface dataset with equal images per class.
    Similar to doggie dataset sampling.
    
    Args:
        base_path: Path to the original RSCD dataset-1million folder
        output_path: Path where sampled dataset will be created
        train_per_class: Number of training images per class
        val_per_class: Number of validation images per class
        test_per_class: Number of test images per class
    """
    base_path = str(base_path)
    output_path = str(output_path)
    
    os.makedirs(output_path, exist_ok=True)
    
    splits = {
        'train': (os.path.join(base_path, 'train'), train_per_class),
        'val': (os.path.join(base_path, 'vali_20k'), val_per_class),
        'test': (os.path.join(base_path, 'test_50k'), test_per_class)
    }
    
    random.seed(42)
    
    for split_name, (split_dir, num_per_class) in splits.items():
        output_split_dir = os.path.join(output_path, split_name)
        os.makedirs(output_split_dir, exist_ok=True)
        
        # Get all class folders
        class_folders = [d for d in os.listdir(split_dir) 
                        if os.path.isdir(os.path.join(split_dir, d))]
        
        print(f"Sampling {split_name} split from {len(class_folders)} classes...")
        
        for class_name in class_folders:
            class_dir = os.path.join(split_dir, class_name)
            output_class_dir = os.path.join(output_split_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Get all images in class folder
            images = [f for f in os.listdir(class_dir) 
                     if os.path.isfile(os.path.join(class_dir, f))]
            
            # Sample random subset
            sampled = random.sample(images, min(num_per_class, len(images)))
            
            # Copy sampled images
            for img in sampled:
                src = os.path.join(class_dir, img)
                dst = os.path.join(output_class_dir, img)
                shutil.copy2(src, dst)
            
            print(f"  {class_name}: sampled {len(sampled)} images")
        
        print(f"Completed {split_name} split in {output_split_dir}")
    
    print(f"\nSampling complete! Output saved to: {output_path}")

def organize_val_images(val_dir):
    """
    Move images in val_dir into subfolders by class, inferred from filename.
    Assumes class name is the prefix before the first underscore or other delimiter.
    """
    for fname in os.listdir(val_dir):
        fpath = os.path.join(val_dir, fname)
        if not os.path.isfile(fpath):
            continue

        # Extract class name from filenames like '202205171728-dry-concrete-smooth.jpg'.
        # The class name is the part after the timestamp, e.g., 'dry-concrete-smooth'.
        base_name = fname.split(' ')[0]
        parts = os.path.splitext(base_name)[0].split('-')
        if len(parts) <= 1:
            continue
        class_name = '-'.join(parts[1:])

        class_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        shutil.move(fpath, os.path.join(class_dir, fname))

    