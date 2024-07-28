import os
import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from KKAN import KKAN
from ConvNet import ConvNet

# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, neg_dir, pos_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Add negative images with label 0
        for filename in os.listdir(neg_dir):
            if os.path.isfile(os.path.join(neg_dir, filename)):
                self.image_paths.append(os.path.join(neg_dir, filename))
                self.labels.append(0)  # Label for negative images

        # Add positive images with label 1
        for filename in os.listdir(pos_dir):
            if os.path.isfile(os.path.join(pos_dir, filename)):
                self.image_paths.append(os.path.join(pos_dir, filename))
                self.labels.append(1)  # Label for positive images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# neg_path = 'Dataset/Gray/Negative'
# pos_path = 'Dataset/Gray/Positive'
neg_path = 'Dataset/Resized/Negative'
pos_path = 'Dataset/Resized/Positive'

full_dataset = ImageDataset(neg_path, pos_path, transform=transform)

# Split the dataset into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 4)

def train(model, device, train_loader, optimizer, epoch, criterion):
    """
    Train the model for one epoch

    Args:
        model: the neural network model
        device: cuda or cpu
        train_loader: DataLoader for training data
        optimizer: the optimizer to use (e.g. SGD)
        epoch: the current epoch
        criterion: the loss function (e.g. CrossEntropy)

    Returns:
        avg_loss: the average loss over the training set
    """

    model.to(device)
    model.train()
    train_loss = 0
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        # Recall that GPU is optimized for the operations we are dealing with
        data, target = data.to(device), target.to(device)
        
        # Reset the optimizer
        optimizer.zero_grad()
        
        # Push the data forward through the model layers
        output = model(data)
        
        # Get the loss
        loss = criterion(output, target)

        # Keep a running total
        train_loss += loss.item()
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    # print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss

def test(model, device, test_loader, criterion):
    """
    Test the model

    Args:
        model: the neural network model
        device: cuda or cpu
        test_loader: DataLoader for test data
        criterion: the loss function (e.g. CrossEntropy)

    Returns:
        test_loss: the average loss over the test set
        accuracy: the accuracy of the model on the test set
        precision: the precision of the model on the test set
        recall: the recall of the model on the test set
        f1: the f1 score of the model on the test set
        confusion_matrix: the confusion matrix of the model on the test set
    """

    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Get the predicted classes for this batch
            output = model(data)
            
            # Calculate the loss for this batch
            test_loss += criterion(output, target).item()
            
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += (target == predicted).sum().item()

            # Collect all targets and predictions for metric calculations
            all_targets.extend(target.view_as(predicted).cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate overall metrics
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')
    confusion_matrix = pd.crosstab(np.array(all_targets), np.array(all_predictions), rownames=['Tacna klasa'], colnames=['Predikcija'])

    # Normalize test loss
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Precision: {:.2f}, Recall: {:.2f}, F1 Score: {:.2f}\n'.format(
    #     test_loss, correct, len(test_loader.dataset), accuracy, precision, recall, f1))

    return test_loss, accuracy, precision, recall, f1, confusion_matrix

def train_and_test_models(model, device, train_loader, test_loader, optimizer, criterion, epochs, scheduler):
    """
    Train and test the model

    Args:
        model: the neural network model
        device: cuda or cpu
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        optimizer: the optimizer to use (e.g. SGD)
        criterion: the loss function (e.g. CrossEntropy)
        epochs: the number of epochs to train
        scheduler: the learning rate scheduler

    Returns:
        all_train_loss: a list of the average training loss for each epoch
        all_test_loss: a list of the average test loss for each epoch
        all_test_accuracy: a list of the accuracy for each epoch
        all_test_precision: a list of the precision for each epoch
        all_test_recall: a list of the recall for each epoch
        all_test_f1: a list of the f1 score for each epoch
    """
    # Track metrics
    all_train_loss = []
    all_test_loss = []
    all_test_accuracy = []
    all_test_precision = []
    all_test_recall = []
    all_test_f1 = []
    all_test_confusion = []
    
    for epoch in range(1, epochs + 1):
        # Train the model
        train_loss = train(model, device, train_loader, optimizer, epoch, criterion)
        all_train_loss.append(train_loss)
        
        # Test the model
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_confusion = test(model, device, test_loader, criterion)
        all_test_loss.append(test_loss)
        all_test_accuracy.append(test_accuracy)
        all_test_precision.append(test_precision)
        all_test_recall.append(test_recall)
        all_test_f1.append(test_f1)
        all_test_confusion.append(test_confusion)

        print(f'End of Epoch {epoch}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2%}')
        scheduler.step()
    model.all_test_accuracy = all_test_accuracy
    model.all_test_precision = all_test_precision
    model.all_test_f1 = all_test_f1
    model.all_test_recall = all_test_recall
    model.all_test_confusion = all_test_confusion

    return all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1, all_test_confusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# time_reps = 10
# elapsed_time = np.zeros(time_reps)
# for i in range(0, time_reps):
#     start_time = time.time()

model_ConvNet = ConvNet()
model_ConvNet.to(device)
optimizer_ConvNet = optim.AdamW(model_ConvNet.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_ConvNet = optim.lr_scheduler.ExponentialLR(optimizer_ConvNet, gamma=0.8)
criterion_ConvNet = nn.CrossEntropyLoss()
all_train_loss_ConvNet, all_test_loss_ConvNet, \
    all_test_accuracy_ConvNet, all_test_precision_ConvNet, \
    all_test_recall_ConvNet, all_test_f1_ConvNet, all_test_confusion_ConvNet = \
        train_and_test_models(model_ConvNet, device, train_loader, test_loader, 
                            optimizer_ConvNet, criterion_ConvNet, epochs=5, scheduler=scheduler_ConvNet)

#     end_time = time.time()
#     elapsed_time[i] = end_time - start_time
#     print(f"Training time: {elapsed_time[i]} s")
# average_time = np.sum(elapsed_time) / time_reps
# print(f"Average ConvNet training time: {average_time} s")

# time_reps = 10
# elapsed_time = np.zeros(time_reps)
# for i in range(0, time_reps):
#     start_time = time.time()

model_KKAN = KKAN(device = device)
model_KKAN.to(device)
optimizer_KKAN = optim.AdamW(model_KKAN.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_KKAN = optim.lr_scheduler.ExponentialLR(optimizer_KKAN, gamma=0.8)
criterion_KKAN = nn.CrossEntropyLoss()
all_train_loss_KKAN, all_test_loss_KKAN, \
    all_test_accuracy_KKAN, all_test_precision_KKAN, \
    all_test_recall_KKAN, all_test_f1_KKAN, all_test_confusion_KKAN = \
        train_and_test_models(model_KKAN, device, train_loader, test_loader, 
                            optimizer_KKAN, criterion_KKAN, epochs=5, scheduler=scheduler_KKAN)

#     end_time = time.time()
#     elapsed_time[i] = end_time - start_time
#     print(f"Training time: {elapsed_time[i]} s")
# average_time = np.sum(elapsed_time) / time_reps
# print(f"Average KKAN training time: {average_time} s")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Plot confusion matrices side by side


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
import seaborn as sns
# Plot confusion matrix for ConvNet
sns.heatmap(all_test_confusion_ConvNet[-1], annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Confusion Matrix - ConvNet')
ax1.set_xlabel('Predikcija')
ax1.set_ylabel('Tacna klasa')

# Plot confusion matrix for KKAN
sns.heatmap(all_test_confusion_KKAN[-1], annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title('Confusion Matrix - KKAN')
ax2.set_xlabel('Predikcija')
ax2.set_ylabel('Tacna klasa')

plt.tight_layout()
plt.show()


def highlight_max(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]

accs = []
precision = []
recall = []
f1s = []
params_counts = []
confs = []

models = [model_ConvNet, model_KKAN]

# Data extraction from models
for i, m in enumerate(models):
    index = np.argmax(m.all_test_accuracy)
    params_counts.append(count_parameters(m))
    accs.append(m.all_test_accuracy[index])
    precision.append(m.all_test_precision[index])
    recall.append(m.all_test_recall[index])
    f1s.append(m.all_test_f1[index])
    confs.append(m.all_test_confusion[index])

# DataFrame creation
df = pd.DataFrame({
    "Test Accuracy": accs,
    "Test Precision": precision,
    "Test Recall": recall,
    "Test F1 Score": f1s,
    "Number of Parameters": params_counts
}, index=[ "ConvNet", "KKAN"])

df.to_csv('surface_cracks.csv', index=False)
df_styled = df.style.apply(highlight_max, subset=df.columns[:], axis=0).format('{:.3f}')

# Inference with timing -----------------------------------------------
# models = {'ConvNet': model_ConvNet, 'KKAN': model_KKAN}
# loader_idx = {'train': train_loader, 'test': test_loader}

# for model_name, model in models.items():
#     for loader_name, loader in loader_idx.items():

#         with torch.no_grad():

#             time_reps = 10
#             elapsed_time = np.zeros(time_reps)
#             for i in range(0, time_reps):
#                 start_time = time.time() 

#                 for data, target in loader:
#                     data, target = data.to(device), target.to(device)
                        
#                     # Get the predicted classes for this batch
#                     output = model(data)

#                 end_time = time.time()
#                 elapsed_time[i] = end_time - start_time
#                 print(f"Inference time: {elapsed_time[i]} s")
#             average_time = np.sum(elapsed_time) / time_reps
#             print(f"Average inference time for model {model_name} on {loader_name} set: {average_time} s")
