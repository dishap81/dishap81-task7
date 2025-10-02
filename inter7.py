# Task 7: Support Vector Machines (SVM) - Simplified

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# 1. Load dataset
file_path = r"C:\Users\Admin\Downloads\task7\breast-cancer.csv"   # <-- Change path if needed
df = pd.read_csv(file_path)

# Drop ID column
df = df.drop(columns=["id"], errors="ignore")

# Encode target (M=1, B=0)
df["diagnosis"] = LabelEncoder().fit_transform(df["diagnosis"])

# Split into features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Standardize features
X_scaled = StandardScaler().fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 2. Linear SVM
svm_linear = SVC(kernel="linear", C=1)
svm_linear.fit(X_train, y_train)
print("Linear SVM Accuracy:", svm_linear.score(X_test, y_test))

# 3. RBF SVM
svm_rbf = SVC(kernel="rbf", C=1, gamma="scale")
svm_rbf.fit(X_train, y_train)
print("RBF SVM Accuracy:", svm_rbf.score(X_test, y_test))

# 4. Hyperparameter tuning
params = {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.1, 1]}
grid = GridSearchCV(SVC(kernel="rbf"), params, cv=5)
grid.fit(X_train, y_train)
print("Best Params:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# 5. Cross-validation
cv_scores = cross_val_score(grid.best_estimator_, X_scaled, y, cv=5)
print("Cross-validation Accuracy:", cv_scores.mean())

# 6. Visualization (PCA 2D)
X_pca = PCA(n_components=2).fit_transform(X_scaled)
svm_vis = SVC(kernel="rbf", C=grid.best_params_["C"], gamma=grid.best_params_["gamma"])
svm_vis.fit(X_pca, y)

# Decision boundary plot
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
plt.title("SVM Decision Boundary (PCA 2D)")
plt.show()