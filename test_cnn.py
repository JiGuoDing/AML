import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger

# Load the trained model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load test data
X_test = pd.read_csv("Dataset/test_image.csv").values.astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 1, 28, 28)  # Reshape for CNN

class TestDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

test_dataset = TestDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model and load weights
model = CNN().to(device)
model.load_state_dict(torch.load("models/best_model_cnn.pth"))
model.eval()

# Make predictions
predictions = []
with torch.no_grad():
    for xb in test_loader:
        xb = xb.to(device)
        outputs = model(xb)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

# Save predictions to CSV
pd.DataFrame({"Label": predictions}).to_csv("test_predictions.csv", index_label="Id")
logger.info("Predictions saved to test_predictions.csv")

# Optional: Print some sample predictions
logger.info(f"First 10 predictions: {predictions[:10]}")