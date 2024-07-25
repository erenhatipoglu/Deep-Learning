import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import CelebA
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from helper_functions import accuracy_fn
import random
import os

# Set environment variable to avoid potential conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Select device for computation
device = "cuda" if torch.cuda.is_available() else -1

# Load CIFAR10 training data
train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None,
    )

# Load CIFAR10 test data
test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)

# Access classes and their indices
train_data.classes
train_data.class_to_idx

# Display shape of a single image and its label
image, label = train_data[0]
image.shape, label

# Create DataLoader for training data
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=32,
                              shuffle=True)

# Create DataLoader for test data
test_dataloader = DataLoader(dataset=train_data,
                             batch_size=32,
                             shuffle=False)

# Training step function
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

# Testing step function
def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

# Evaluate model function
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):

    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

# Define the CNN model
class CIFAR10model_Conv(nn.Module):

    def __init__(self, input_shape:int, hidden_layers:int, output_shape:int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=hidden_layers,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Conv2d(in_channels=hidden_layers,
                      out_channels=hidden_layers,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
            )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_layers, hidden_layers*2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_layers*2, hidden_layers*2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_layers*2*8*8,
                      out_features=output_shape)
        )

    def forward(self,x:torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x= self.classifier(x)
        return x

# Set random seed for reproducibility
torch.manual_seed(42)

# Instantiate the model
model = CIFAR10model_Conv(input_shape=3,
                          hidden_layers=10,
                          output_shape=len(train_data.classes)).to(device)

model

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), 
                             lr=0.01)

# Set random seed for reproducibility
torch.manual_seed(42)

# Measure the start time
start = time.time()

# Set number of epochs
epochs = 50

# Training loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}")

    train_step(data_loader=train_dataloader,
               model=model,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device
              )
    test_step(data_loader=test_dataloader,
              model=model,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device
              )
    
# Measure the end time
end = time.time()

# Calculate total training time
total_train_time = start-end
print(f"Total train time: {abs(total_train_time)} seconds.")

# Function to make predictions
def make_predictions(model:torch.nn.Module,data:list,device:torch.device=device):
    pred_probs = list()
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample,dim=0).to(device)

            pred_logit = model(sample)

            pred_prob = torch.softmax(pred_logit.squeeze(),dim=0)

            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)

# Number of rows and columns for plotting
nrows = 5
ncols = 5

# Set random seed for reproducibility
random.seed(42)
test_samples = list()
test_labels = list()

# Select random samples from test data
for sample, label in random.sample(list(test_data), k=nrows*ncols):
    test_samples.append(sample)
    test_labels.append(label)

# Make predictions
pred_probs = make_predictions(model=model, data=test_samples)
pred_classes = pred_probs.argmax(dim=1)

# Plot the images with predictions
plt.figure(figsize=(15, 15))

for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i + 1)
    sample = sample.permute(1, 2, 0).numpy()  # Convert to (height, width, channels)
    sample = (sample * 255).astype(np.uint8)  # Convert to 0-255 range
    plt.imshow(sample)
    pred_label = train_data.classes[pred_classes[i]]
    truth_label = train_data.classes[test_labels[i]]
    title_text = f"Pred: {pred_label} | Reality: {truth_label}"
    if pred_label == truth_label:
        plt.title(title_text, fontsize=8, color="g")
    else:
        plt.title(title_text, fontsize=8, color="r")
    plt.axis('off')

plt.tight_layout()
plt.show()
