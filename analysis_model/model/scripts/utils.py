import main_model
import augmentation
from main_model import GrubDataset
from main_model import DataLoader

import glob

# Initializing batch size
n = 32

# Train dataset and initialize dataloader
train_path = "c:/Users/pjeni/pyfileshare/GatorGrubAI/analysis_model/data/train_images"
valid_path = "c:/Users/pjeni/pyfileshare/GatorGrubAI/analysis_model/data/valid_images"

train_dataset = GrubDataset(train_path)
print(f"Number of training images: {len(train_dataset)}")
train_loader = DataLoader(train_dataset, batch_size=n, shuffle=True)
train_N = len(train_loader.dataset)

# Validation dataset and initialize dataloader
valid_dataset = GrubDataset(valid_path)
print(f"Number of validation images: {len(valid_dataset)}")
valid_loader = DataLoader(valid_dataset, batch_size=n, shuffle=False)
valid_N = len(valid_loader.dataset)

import torch

def train(model, train_loader, train_N, optimizer, loss_function):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    correct = 0

    # Get the device from the model's parameters
    device = next(model.parameters()).device

    for images, labels in train_loader:
        # Move data to the appropriate device (CPU or GPU)
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_function(outputs, labels)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / train_N
    print(f"Train - Loss: {avg_loss:.4f} Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

def validate(model, valid_loader, valid_N, loss_function):
    """
    Validate the model on the validation dataset.
    """
    model.eval()
    total_loss = 0
    correct = 0

    # Get the device from the model's parameters
    device = next(model.parameters()).device

    with torch.no_grad():
        for images, labels in valid_loader:
            # Move data to the appropriate device (CPU or GPU)
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()

            # Compute accuracy
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(valid_loader)
    accuracy = correct / valid_N
    print(f"Valid - Loss: {avg_loss:.4f} Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy