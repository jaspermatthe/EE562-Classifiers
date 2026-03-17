import torch
import torch.nn as nn  
import torch.optim as optim  
from torchvision import models  
import sys
from tqdm import tqdm 
import numpy as np
project_root = "/Users/cindychen/Desktop/EE562/EE562 Assignments/EE562-Classifiers"
if project_root not in sys.path:
    sys.path.append(project_root)

from src.datasets.doggie_loader import get_doggie_dataset, create_doggie_dataloaders, get_default_transforms
# resnet-18 classifier
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.backbone = models.resnet18(weights=None) # architecture with random weights (not pretrained)
        num_features = self.backbone.fc.in_features # getting the number of input features to the original fully connected layer, 
        # we need this number to connect our new classifier head, because this one is assuming the same number of input features as the original ResNet-18, which is 512. 
        self.backbone.fc = nn.Sequential( #the original classifier is designed for 1000 ImageNet classes, we need to adapt it to our specific datasets.
            nn.Linear(num_features, 512),# First fully connected layer: from num_features -> 512, depending on the dataset
            nn.ReLU(), # RELU for non-linearity
            nn.Dropout(0.5), # dropout to prevent overfitting, 50% rate
            nn.Linear(512, num_classes)  # Final classification layer: 512 -> num_classes
        )
    
    # forward pass through the network
    def forward(self, x):
        return self.backbone(x)

# creating the model, no frozen layers, we want to train the entire network from scratch, so we initialize it with random weights (weights=None) and we will train all layers on our dataset.
def create_resnet18(num_classes):
    model = ResNetClassifier(num_classes=num_classes) # num classes are the number of output classes
    return model


# training model for one epoch, this function will be called in a looop for multiple epochs
def train_epoch(model, train_loader, criterion, optimizer, device='cpu'):
    model.train() # model set to training mode
    # Initialize accumulators for loss and accuracy
    running_loss = 0.0 
    correct = 0         
    total = 0         
    
    # create progress bar for better visualization
    pbar = tqdm(train_loader, desc='Training') #train loader is the dataloader for the training data 
    # iterating through batches of data
    for inputs, labels in pbar:
        optimizer.zero_grad() #zero the parameter gradients, otherwise gradients would accumulate across batches
        outputs = model(inputs) # forward pass, predictions for the current batch
        loss = criterion(outputs, labels)# Calculate loss between predictions and true labels
        loss.backward() # backwards pass, computing graidents
        optimizer.step() # updating model weights
        running_loss += loss.item() * inputs.size(0) # accumulate the losses, multiplying by batch size to get the total loss for the batch, which will be averaged later over the entire dataset.
        _, predicted = torch.max(outputs, 1) # find the predicted class, which is the index with the highest score
        #torch.max returns (values, indices), we want the indices
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
        pbar.set_postfix({ # update progress bar with current loss and accuracy
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100 * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader.dataset) # average loss calculation
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# validation function, similar to training but without backpropagation and weight updates, and with model set to evaluation mode.
def validate_epoch(model, val_loader, criterion):
    model.eval() # evaluation mode, without dropout and batchnorm updates
    running_loss = 0.0
    correct = 0
    total = 0
    # no more gradient computations for validation, which saves memory and speeds up computations
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for inputs, labels in pbar:
            outputs = model(inputs) # forward pass to get predictions for the validation batch, no backprop or weight updates
            loss = criterion(outputs, labels) # the rest is the same as training
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# prediction function for a single image as an example in the report
def predict_image(image_path, model, class_names, transform):
    from PIL import Image
    model.eval()
    image = Image.open(image_path).convert('RGB') # open the image and conver to RGB
    image = transform(image).unsqueeze(0) # apply transform and add batch dimensions at position 0
    with torch.no_grad(): # disable gradient computation
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1) # convert to probabilieis that sum up to 1
        predicted_class_idx = torch.argmax(probabilities, dim=1).item() 
        confidence = probabilities[0][predicted_class_idx].item() # confidence score for the predicted class
    
    return class_names[predicted_class_idx], confidence

# get predictions for al the samples in a dataloader, this is useful for evaluating the model on the entire validation set
def get_all_predictions(model, loader):
    model.eval()
    all_preds = [] # for storing predicted class indices
    all_labels = []

    with torch.no_grad(): # no gradient computation
        # Iterate through all batches
        for inputs, labels in loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.numpy()) # numpy array of predicted class indices
            all_labels.extend(labels.numpy()) # ground truth labels

    return np.array(all_preds), np.array(all_labels) # return numpy arrays of predictions and labels for further analysis (e.g., confusion matrix, classification report)