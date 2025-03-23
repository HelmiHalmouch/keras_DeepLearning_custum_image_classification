import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

class CustomImageClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def preprocess_data(data_path='data', img_rows=128, img_cols=128, use_sklearn_preprocessing=False):
    PATH = os.getcwd()
    data_path = os.path.join(PATH, data_path)
    data_dir_list = os.listdir(data_path)
    img_data_list = []
    labels = []

    for idx, dataset in enumerate(data_dir_list):
        img_list = os.listdir(os.path.join(data_path, dataset))
        for img in img_list:
            input_img = cv2.imread(os.path.join(data_path, dataset, img))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_resize = cv2.resize(input_img, (img_rows, img_cols))
            img_data_list.append(input_img_resize)
            labels.append(idx)

    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32') / 255.0  # Normalize the images

    # Add channel dimension (N, C, H, W)
    img_data = np.expand_dims(img_data, axis=1)  # Single channel grayscale images

    if use_sklearn_preprocessing:
        img_data_scaled = preprocessing.scale(img_data.reshape(len(img_data), -1))
        img_data_scaled = img_data_scaled.reshape(len(img_data), 1, img_rows, img_cols)
        img_data = img_data_scaled

    labels = np.array(labels)
    return torch.tensor(img_data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def prepare_data_loaders(X_data, Y_data, batch_size=16, test_size=0.2):
    X_data, Y_data = shuffle(X_data, Y_data, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=test_size, random_state=42)

    train_data = torch.utils.data.TensorDataset(X_train, Y_train)
    test_data = torch.utils.data.TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_test, Y_test

def train_model(model, train_loader, test_loader, num_epoch=50, optimizer=None, criterion=None, callbacks=None):
    # TensorBoard setup
    writer = SummaryWriter(log_dir='runs/CustomImageClassifier')

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total

        # Validation after each epoch
        model.eval()
        val_loss_epoch = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss_epoch += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss_epoch /= len(test_loader)
        val_accuracy_epoch = val_correct / val_total

        # Save best model
        if val_accuracy_epoch > best_acc:
            best_acc = val_accuracy_epoch
            best_model_wts = model.state_dict()

        # TensorBoard Logging
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_accuracy, epoch)
        writer.add_scalar('Validation/Loss', val_loss_epoch, epoch)
        writer.add_scalar('Validation/Accuracy', val_accuracy_epoch, epoch)

        print(f"Epoch [{epoch + 1}/{num_epoch}], "
              f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, "
              f"Val Loss: {val_loss_epoch:.4f}, Val Accuracy: {val_accuracy_epoch:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model

def make_predictions(model, X_test, Y_test):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test).to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(Y_test.numpy(), predicted.cpu().numpy())
        print(f"Test Accuracy: {accuracy:.4f}")
    return predicted.cpu().numpy()

def save_model(model, model_path='model.pth'):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def plot_results(train_loss, train_accuracy, val_loss, val_accuracy):
    plt.figure(0)
    plt.plot(train_accuracy, 'r', label='Train Accuracy')
    plt.plot(val_accuracy, 'g', label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig('accuracy_plot.png')

    plt.figure(1)
    plt.plot(train_loss, 'r', label='Train Loss')
    plt.plot(val_loss, 'g', label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig('loss_plot.png')


# Usage example
if __name__ == "__main__":
    # Hyperparameters and device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    X_data, Y_data = preprocess_data(data_path='data')
    train_loader, test_loader, X_test, Y_test = prepare_data_loaders(X_data, Y_data)

    # Initialize model, criterion, optimizer
    model = CustomImageClassifier(num_classes=4).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train model
    model = train_model(model, train_loader, test_loader, num_epoch=50, optimizer=optimizer, criterion=criterion)

    # Save model
    save_model(model)

    # Make predictions on X_test
    predicted_labels = make_predictions(model, X_test, Y_test)
