import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.Sigmoid(),
            nn.Linear(128, 10)
        )
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    def forward(self, x):
        return self.model(x)

# Hyperparameters
epoch = 10
batch_size = 256
learning_rate = 0.05
momentum = 0.9

# Load and preprocess data
train_data = np.load('./data/train_data.npy')
train_label = np.load('./data/train_labels.npy')
train_data = train_data / 255.0 - 0.5
train_label = train_label.astype(np.int32)

test_data = np.load('./data/test_data.npy')
test_label = np.load('./data/test_labels.npy')
test_data = test_data / 255.0 - 0.5
test_label = test_label.astype(np.int32)

# Convert to PyTorch tensors
train_data = torch.tensor(train_data, dtype=torch.float32)
train_label = torch.tensor(train_label, dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.float32)
test_label = torch.tensor(test_label, dtype=torch.long)

# Create data loaders
train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Training loop
for e in range(epoch):
    print(f"Epoch {e + 1} started")
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
    print(f"Epoch {e + 1} finished, Average Loss: {total_loss / batch_count}")
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    print(f"Epoch {e + 1} finished, Test accuracy: {accuracy * 100:.2f}%")