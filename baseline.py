import pandas as pd

from sklearn.model_selection import train_test_split
import numpy as np
import copy
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

#一个例子，可自行处理数据，自行划分验证集

class CustomDataset():
    def __init__(self, file_path, selected_columns=None, is_label=False):
        self.data = pd.read_csv(file_path)  # 读取CSV文件
        self.selected_columns = selected_columns
        self.is_label = is_label
        if(self.selected_columns == None):
            self.selected_columns = self.data.columns
        if self.is_label:
            self.label_column = self.selected_columns[0]
        
        self.Handling_missing_values()

        # print("-"*50)
        # print("selected_columns : ",self.selected_columns)
        # print("DataFrame.column : ",self.data.columns)
    
    def Handling_missing_values(self):
        self.data = self.data.dropna()

    def print(self):
        dataset = self.data[self.selected_columns]
        print(dataset)

    def normalized(self):
        legal_columns = copy.deepcopy(self.selected_columns)
        mean = self.data[legal_columns].mean()
        std = self.data[legal_columns].std()
        self.data[legal_columns] = (self.data[legal_columns] - mean)/std

    def getx(self):
        legal_columns = copy.deepcopy(self.selected_columns)
        return self.data[legal_columns].to_numpy()
    
    def gety(self):
        return self.data[self.label_column].to_numpy()
    
    def __len__(self):
        return len(self.data)


unlabel_train_dataset_x = CustomDataset('Dataset/train_image_unlabeled.csv')
train_dataset_x = CustomDataset('Dataset/train_image_labeled.csv')
train_dataset_y = CustomDataset('Dataset/train_label.csv', is_label=True)


X_train, y_train, unlabel_X_train = train_dataset_x.getx(), train_dataset_y.gety(), unlabel_train_dataset_x.getx()
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42,shuffle=True)
for i in range(10):
    count = np.count_nonzero(y_train == i)
    print(f"Train : {i} 类别个数：",count)


def eval(y_pred,y_true):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')    #micro_f1 恒等 acc
    print("acc : ",accuracy)
    print("macro f1 :", macro_f1)

    
    
# randomforest
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train, y_train)
print(X_valid.shape, y_valid.shape)
y_pred_rf_valid = clf_rf.predict(X_valid)
print("---------- Randomforest Valid Eval ----------")
eval(y_pred_rf_valid, y_valid)


# MLP
clf_mlp = MLPClassifier()
clf_mlp.fit(X_train, y_train)
y_pred_rf_valid = clf_mlp.predict(X_valid)
print("---------- MLP Valid Eval ----------")
eval(y_pred_rf_valid, y_valid)

# GBDT
clf_gbdt = GradientBoostingClassifier()
clf_gbdt.fit(X_train, y_train)
y_pred_rf_valid = clf_gbdt.predict(X_valid)
print("---------- GBDT Valid Eval ----------")
eval(y_pred_rf_valid, y_valid)

#Decision Tree
clf = DecisionTreeClassifier()
clf_dt = clf.fit(X_train, y_train)
y_pred_rf_valid = clf_dt.predict(X_valid)
print("---------- Decision Tree Valid Eval ----------")
eval(y_pred_rf_valid, y_valid)


