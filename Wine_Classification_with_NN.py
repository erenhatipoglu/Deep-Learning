   import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import requests
from pathlib import Path 

wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)

NUM_CLASSES = 3
NUM_FEATURES = 13
SEED = 42

# Convert the dataset to numpy arrays
X_wine = wine_df.values
y_wine = wine.target

# Convert numpy arrays to PyTorch tensors
X_wine = torch.from_numpy(X_wine).type(torch.float)
y_wine = torch.from_numpy(y_wine).type(torch.LongTensor)

X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, train_size=0.8, random_state=SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Move data to the device
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# Plot the dataset
plt.figure(figsize=(10,7))
plt.scatter(X_wine[:,0], X_wine[:,1], c=y_wine, cmap=plt.cm.RdYlBu)
plt.show()

# Define the neural network model
class WineModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_layers=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_layers),
            nn.Sigmoid(),
            nn.Linear(in_features=hidden_layers, out_features=hidden_layers),
            nn.Sigmoid(),
            nn.Linear(in_features=hidden_layers, out_features=output_features)
        )
    
    def forward(self, x):
        return self.linear_layer_stack(x)

# Instantiate the model
wine_model = WineModel(input_features=NUM_FEATURES, output_features=NUM_CLASSES, hidden_layers=8).to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=wine_model.parameters(), lr=0.01)

# Training loop
torch.manual_seed(101)
epochs = 100

for epoch in range(epochs):
    wine_model.train()

    # Forward pass
    y_logits = wine_model(X_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    # Calculate the loss
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_score(y_train.cpu(), y_pred.cpu())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print training progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {loss:.5f}, Acc {acc:.2f}")

# Testing
with torch.no_grad():
    wine_model.eval()
    test_logits = wine_model(X_test)
    test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
    test_loss = loss_fn(test_logits, y_test)
    test_acc = accuracy_score(y_test.cpu(), test_pred.cpu())

    print(f"Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}")
