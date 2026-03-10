import random, shutil
from pathlib import Path

# function to split the stanford dogs dataset into the same ratio's of train, validate and test as the other dataset
def split_stanford_dogs(source="/Users/cindychen/Desktop/EE562/EE562 Assignments/EE562-Classifiers/data/Images", output="/Users/cindychen/Desktop/EE562/EE562 Assignments/EE562-Classifiers/data/dataset_split", train_ratio=0.8, val_ratio=0.1, test_ratio=1, seed=42):
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
        
        for split_name, split_imgs in [
            ('train', images[:n_train]),
            ('val', images[n_train:n_train + n_val]),
            ('test', images[n_train + n_val:])
        ]:
            dest_dir = output / split_name / class_folder.name
            dest_dir.mkdir(exist_ok=True)
            for img in split_imgs:
                shutil.copy(img, dest_dir / img.name)
        
        print(f"{class_folder.name}: {n_train} train, {n_val} val, {n - n_train - n_val} test")


