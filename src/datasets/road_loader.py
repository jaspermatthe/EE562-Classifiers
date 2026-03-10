from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import shutil

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

    