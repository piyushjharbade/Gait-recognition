import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


# Generate Double Moon Dataset
def generate_double_moon(N=1000, radius=10, width=2, d=1.5):
    """
    Generates two interleaving half circles (moons).
    radius: distance from center to arc
    width: thickness of the moon
    d: vertical shift of lower moon (positive = no overlap)
    """
    N1 = N // 2

    # Upper moon (class 0)
    theta1 = np.random.uniform(0, np.pi, N1)
    x1 = radius * np.cos(theta1)
    y1 = radius * np.sin(theta1)
    x1 += np.random.uniform(-width / 2, width / 2, N1)
    y1 += np.random.uniform(-width / 2, width / 2, N1)
    upper = np.stack((x1, y1), axis=1)

    # Lower moon (class 1)
    theta2 = np.random.uniform(0, np.pi, N1)
    x2 = radius * np.cos(theta2)
    y2 = -radius * np.sin(theta2) + d
    x2 += np.random.uniform(-width / 2, width / 2, N1)
    y2 += np.random.uniform(-width / 2, width / 2, N1)
    x2 = x2 + radius
    y2 = y2 - d
    lower = np.stack((x2, y2), axis=1)

    data = np.vstack((upper, lower)).astype(np.float32)
    labels = np.hstack((np.zeros(N1), np.ones(N1))).astype(np.float32)
    return data, labels


# MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[10, 10]):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.network(x))

    # Training function (returns loss history)


def train(model, criterion, optimizer, X_train, y_train, epochs=100):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        inputs = torch.from_numpy(X_train)
        labels = torch.from_numpy(y_train).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    return loss_history


# Evaluation
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X_test)
        labels = torch.from_numpy(y_test).unsqueeze(1)
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        acc = (predicted.eq(labels).sum().item()) / len(y_test)
        print(f"Accuracy: {acc * 100:.2f}%")

    # Decision Boundary Plot


def plot_decision_boundary(model, X, y):
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    with torch.no_grad():
        Z = model(grid)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.6, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title("Decision Boundary (PyTorch MLP)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.show()


# Plot Loss Curve
def plot_loss_curve(loss_history):
    plt.figure()
    plt.plot(loss_history, label='Loss')


plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.legend()
plt.show()
# Main
if __name__ == "__main__":
    # Generate data
    X, y = generate_double_moon(N=2000, radius=10, width=6, d=-3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Model setup
    model = MLPClassifier(input_dim=2, hidden_dims=[10, 10])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # train and plot
    loss_history = train(model, criterion, optimizer, X_train, y_train, epochs=200)
    evaluate(model, X_test, y_test)
    plot_decision_boundary(model, X, y)
    plot_loss_curve(loss_history)
