import torch
import torch.nn as nn
import pandas as pd

# 1. 加载模型（确保模型定义与你保存的模型一致）
class MyMNISTModel(nn.Module):
    def __init__(self):
        super(MyMNISTModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

model = MyMNISTModel()
model.load_state_dict(torch.load("models/model.pt"))
model.eval()

# 2. 读取未标注数据
df = pd.read_csv("Dataset/train_image_unlabeled.csv")
data = torch.tensor(df.values, dtype=torch.float32)

# 3. 归一化（如果训练时有做，就要做）
data /= 255.0

# 4. 预测
with torch.no_grad():
    outputs = model(data)
    predictions = torch.argmax(outputs, dim=1)

# 5. 保存预测结果
pd.DataFrame(predictions.numpy(), columns=["label"]).to_csv("predictions.csv", index=False)