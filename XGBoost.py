import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


dataset_files = ['dataset_apple.xlsx', 'dataset_corn.xlsx', 'dataset_grape.xlsx', 'dataset_potato.xlsx','dataset_potato.xlsx']
datasets = []
for file in dataset_files:
    dataset = pd.read_excel(file)
    datasets.append(dataset)


combined_dataset = pd.concat(datasets, ignore_index=True)


correlation_matrix = combined_dataset.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


high_correlation_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.97:
            colname = correlation_matrix.columns[i]
            high_correlation_features.add(colname)
            
combined_dataset.drop(high_correlation_features, axis=1, inplace=True)


X = combined_dataset.drop('label', axis=1)
y = combined_dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


xgb_model = XGBClassifier(learning_rate=0.2, n_estimators=300,max_depth=10)
xgb_model.fit(X_train, y_train)


y_pred = xgb_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
