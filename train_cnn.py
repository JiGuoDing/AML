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
X_labeled = pd.read_csv("Dataset/train_image_labeled.csv").values.astype(np.float32) / 255.0
y_labeled = pd.read_csv("Dataset/train_label.csv").values.flatten().astype(np.int64)
X_unlabeled = pd.read_csv("Dataset/train_image_unlabeled.csv").values.astype(np.float32) / 255.0

# 转换为 CNN 输入格式
X_labeled = X_labeled.reshape(-1, 1, 28, 28)
X_unlabeled = X_unlabeled.reshape(-1, 1, 28, 28)

# 划分有标签数据的训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)

class MNISTDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
        self.is_labeled = y is not None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.is_labeled:
            return self.X[idx], self.y[idx]
        return self.X[idx]

train_labeled_dataset = MNISTDataset(X_train, y_train)
val_dataset = MNISTDataset(X_val, y_val)
unlabeled_dataset = MNISTDataset(X=X_unlabeled)

train_labeled_loader = DataLoader(train_labeled_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64)

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

# 训练参数
epochs = 100
# 无标签数据损失的权重
alpha = 0.5
# 置信度阈值，只有高于这个阈值的预测才会用于训练
threshold = 0.95

# 训练并验证
for epoch in range(epochs):
    # 训练
    model.train()
    total_loss = 0.0
    
    # 先训练有标签数据
    for xb, yb in train_labeled_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 再使用无标签数据生成伪标签并训练
    model.eval()
    pseudo_labels = []
    pseudo_data = []
    
    with torch.no_grad():
        for xb in unlabeled_loader:
            xb = xb.to(device)
            outputs = model(xb)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            
            # 只选择高置信度的样本
            mask = confidences > threshold
            if mask.sum() > 0:
                pseudo_data.append(xb[mask])
                pseudo_labels.append(preds[mask])
    
    # 如果有高置信度的伪标签样本，则进行训练
    if len(pseudo_data) > 0:
        pseudo_data = torch.cat(pseudo_data)
        pseudo_labels = torch.cat(pseudo_labels)
        
        model.train()
        for i in range(0, len(pseudo_data), 64):
            xb = pseudo_data[i:i+64]
            yb = pseudo_labels[i:i+64]
            
            pred = model(xb)
            loss = alpha * loss_fn(pred, yb)  # 使用较小的权重
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

    # 验证阶段
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
    avg_loss = total_loss / (len(train_labeled_loader) + (len(pseudo_data) // 64 + 1 if len(pseudo_data) > 0 else 0))
    
    logger.info(f"\nEpoch {epoch+1} - Loss: {avg_loss:.4f} - Accuracy: {acc:.4f}")
    logger.info(classification_report(y_true, y_pred, digits=4))

    # 保存最优模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), save_path)
        logger.info(f"✅ New best model saved at epoch {epoch+1} with accuracy {best_acc:.4f}")
