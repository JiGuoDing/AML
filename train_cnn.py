import pandas as pd
import torch
from loguru import logger
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 加载数据集
X = pd.read_csv("Dataset/train_image_labeled.csv").values.astype(np.float32) / 255.0
y = pd.read_csv("Dataset/train_label.csv").values.flatten().astype(np.int64)
# 转换为 CNN 输入格式
X = X.reshape(-1, 1, 28, 28)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

class MNISTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = MNISTDataset(X_train, y_train)
val_dataset = MNISTDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# CNN
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

model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 初始化
best_acc = 0.0
save_path = "models/best_model_cnn.pth"

# 训练并验证
for epoch in range(100):
    # 训练
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            pred = model(xb).argmax(1).cpu()
            y_pred.extend(pred.numpy())
            y_true.extend(yb.numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"\nEpoch {epoch+1} Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))

    # 保存最优模型
    if acc > best_acc:
        best_acc = acc
        # torch.save(model.state_dict(), save_path)
        logger.info(f"✅ New best model saved at epoch {epoch+1} with accuracy {best_acc:.4f}")
