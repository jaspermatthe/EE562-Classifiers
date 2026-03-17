import random, shutil
import sys
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
from pathlib import Path

project_root = "/Users/cindychen/Desktop/EE562/EE562 Assignments/EE562-Classifiers"
if project_root not in sys.path:
    sys.path.append(project_root)

# function to split the stanford dogs dataset into the same ratio's of train, validate and test as the other dataset
def split_stanford_dogs(source="/Users/cindychen/Desktop/EE562/EE562 Assignments/EE562-Classifiers/data/Images", 
                        output="/Users/cindychen/Desktop/EE562/EE562 Assignments/EE562-Classifiers/data/dataset_split", 
                        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    source, output = Path(source), Path(output)
    
    random.seed(seed)
    for split in ['train', 'val', 'test']:
        (output / split).mkdir(parents=True, exist_ok=True)
    
    for class_folder in source.iterdir():
        if not class_folder.is_dir():
            continue
            
        images = list(class_folder.glob('*.jpg'))
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = [
            ('train', images[:n_train]),
            ('val', images[n_train:n_train + n_val]),
            ('test', images[n_train + n_val:])
        ]
        
        for split_name, split_imgs in splits:
            dest_dir = output / split_name / class_folder.name
            dest_dir.mkdir(exist_ok=True)
            for img in split_imgs:
                shutil.copy(img, dest_dir / img.name)
        
        print(f"{class_folder.name}: {n_train} train, {n_val} val, {n - n_train - n_val} test")
    print(f"\nDataset split complete! Files saved to: {output}")
    return output

# # defining the default data transformations for training and validation.
# def get_default_transforms():
#     # training transforms will be different from validation transforms, because we're applying data augmentation on training transforms.
#     train_transform = transforms.Compose([
#         transforms.Resize((224, 224)),# Resize to 224x224 as expected by resnet-18
#         transforms.RandomHorizontalFlip(p=0.5), # randomly flip the images horizontally with a 50% chance
#         transforms.RandomRotation(15), # randomly rotating images by up to 15 degrees
#         transforms.ColorJitter( # randomly changing the brightness, contrast and saturation of the images.
#             brightness=0.2, 
#             contrast=0.2, 
#             saturation=0.2
#         ),
#         transforms.ToTensor(),
#         transforms.Normalize( # normalization with imagenet statistics
#             mean=[0.485, 0.456, 0.406],  # RGB means from ImageNet
#             std=[0.229, 0.224, 0.225]     # RGB stds from ImageNet
#         )
#     ])
    
#     # we want to evaluate on clean, unmodified images to get true performance, so val doesn't have augmentation
#     val_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])
    
#     return train_transform, val_transform

# more aggresssive data augmentation for training, to help the model generalize better given the small dataset size, and to make it more robust to variations in the images.
def get_default_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize slightly larger then crop
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random scale and crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-30, 30)),  # Increased rotation range
        transforms.ColorJitter(  # More aggressive color jitter
            brightness=0.3, 
            contrast=0.3, 
            saturation=0.3,
            hue=0.1  # Added hue jitter
        ),
        transforms.RandomAffine(  # NEW: Random affine transformations
            degrees=0,  # Already using rotation above
            translate=(0.1, 0.1),  # Random shift up to 10%
            scale=(0.9, 1.1),  # Random scaling
            shear=10  # Random shear
        ),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # NEW: Random blur
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),  # NEW: Random sharpness
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # NEW: Cutout regularization
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # VALIDATION/TEST TRANSFORMS - Keep simple
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),  # Center crop instead of direct resize
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform

def get_doggie_dataset(data_path="/Users/cindychen/Desktop/EE562/EE562 Assignments/EE562-Classifiers/data/dataset_split"):
    data_path = Path(data_path)
    train_transform, val_transform = get_default_transforms() # get the transforms
    #loading datasets
    train_dataset = datasets.ImageFolder(root=data_path/'train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root=data_path/'val', transform=val_transform)
    test_dataset = datasets.ImageFolder(root=data_path/'test', transform=val_transform)
    #getting the class information
    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}") 
    print(f"Test samples: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, class_names, num_classes

def create_doggie_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16, num_workers=0):
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    print(f"\nUsing batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    split_stanford_dogs()