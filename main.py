import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset from a file in a folder
df_loan = pd.read_csv('C:\\Users\\lenovo\\Documents\\UCF\\Spr24\\CAP5619\\Data\\load_data.csv',
                      header=None)

# Dataset has columns 'Class label', 'Feature1', 'Feature2', etc.
# Adjust the column names accordingly based on your actual dataset
df_loan.columns = ['Class label', 'ID', 'Gender', 'loan_amount',
                   'income', 'Credit_Score', 'Status']

df_loan = df_loan[df_loan['Class label'] != 1]

y = df_loan['Class label']
X = df_loan[['ID', 'Status']]

le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and testing sets
# Train the classifier on the training data
X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                     test_size=0.2,
                     random_state=1,
                     stratify=y)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=1)

# Make predictions on the test data
tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

x_min = X_train.values[:, 0].min() - 1
x_max = X_train.values[:, 0].max() + 1
y_min = X_train.values[:, 1].min() - 1
y_max = X_train.values[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(nrows=1, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(8, 3))

for idx, clf, tt in zip([0, 1], [tree], ['Decision tree']):
    clf.fit(X_train, y_train)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='green', marker='o')
    axarr[idx].set_title(tt)

axarr[0].set_ylabel('Status', fontsize=12)

plt.tight_layout()
plt.text(0, -0.2, s='Gender', ha='center', va='center', fontsize=12, transform=axarr[1].transAxes)
plt.show()
