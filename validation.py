import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models, datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directories
data_root_dir = "/content/drive/MyDrive/Colab Notebooks/TrashType_Image_Dataset"

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load the full dataset first
full_dataset = datasets.ImageFolder(root=data_root_dir, transform=transform)
num_classes = len(full_dataset.classes)
print(f"Detected {num_classes} classes: {full_dataset.classes}")

# Define the split ratios
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(full_dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# Load pre-trained model
model = models.mobilenet_v2(pretrained=True)
if isinstance(model, models.MobileNetV2):
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
else:
    # Fallback for other models, might need adjustment
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        last_layer = model.classifier[-1]
        if isinstance(last_layer, nn.Linear):
            model.classifier[-1] = nn.Linear(last_layer.in_features, num_classes)
        else:
            print("Warning: Could not automatically replace the last layer of the classifier. Please inspect model structure.")
    else:
        print("Warning: Model classifier replacement might be incorrect. Please check the model's architecture.")

model = model.to(device)
print(f"Classifier replaced. Model moved to {device}.")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop with Training Accuracy
def train_model(model, epochs=5):
    model.train()
    print("Starting training...")
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy for the current batch
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        epoch_time = time.time() - start_time
        train_accuracy = 100 * correct_train / total_train
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {running_loss/len(train_loader):.4f} - Train Accuracy: {train_accuracy:.2f}% - Time: {epoch_time:.2f}s")
    print("Training finished.")

# Evaluate (your existing function, renamed for clarity)
def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0
    print("Starting validation evaluation...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy

# Run training and evaluation
if __name__ == '__main__':
    start_total_time = time.time()
    train_model(model, epochs=5)
    evaluate_model(model)
    end_total_time = time.time()
    print(f"Total execution time: {(end_total_time - start_total_time):.2f}s")