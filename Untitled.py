#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np

Churn_model = pd.read_csv('Churn_Modelling.csv')
Churn_model.head()
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

Churn_model['NumOfProducts'] = pd.Categorical(Churn_model.NumOfProducts)
Churn_model['HasCrCard'] = pd.Categorical(Churn_model.HasCrCard)
Churn_model['IsActiveMember'] = pd.Categorical(Churn_model.IsActiveMember)
Churn_model['Tenure'] = pd.Categorical(Churn_model.Tenure)
Churn_model['Gender'] = pd.Categorical(Churn_model.Gender)
Churn_model['Exited'] = pd.Categorical(Churn_model.Exited)

Churn_model.describe(include='all')

print(Churn_model['NumOfProducts'].value_counts())
print(Churn_model['HasCrCard'].value_counts())
print(Churn_model['IsActiveMember'].value_counts())
print(Churn_model['Exited'].value_counts())
print(Churn_model['Tenure'].value_counts())
print(Churn_model['Geography'].value_counts())
print(Churn_model['Gender'].value_counts())

print(Churn_model['Age'].value_counts())

dataset = Churn_model
x = dataset.data
y = dataset.target

cut_labels_4 = ['1', '2', '3', '4', '5', '6']
cut_bins = [17, 30, 40, 50, 60, 70, 95]
Churn_model['Age_Bin'] = pd.cut(Churn_model['Age'], bins=cut_bins, labels=cut_labels_4)
print(Churn_model['Age_Bin'].value_counts())


def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
    self.lr = learning_rate
    self.lambda_param = lambda_param
    self.n_iters = n_iters
    self.w = None
    self.b = None


def fit(self, X, y):
    y_ = np.where(y <= 0, -1, 1)
    n_samples, n_features = X.shape

    self.w = np.zeros(n_features)
    self.b = 0

    for _ in range(self.n_iters):

        condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
        if condition:
            self.w -= self.lr * (2 * self.lambda_param * self.w)
        else:
            self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))


X, y = datasets.make_blobs(n_samples=50, n_features=3, centers=3, cluster_std=1.05, random_state=40)
y = np.where(y == 0, -1, 1)

clf = SVC()
clf.fit(X, y)
predictions = clf.predict(X)

print(clf.w, clf.b)


def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k')

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])


Churn_model = pd.get_dummies(Churn_df,
                             prefix=['Geo', 'Gen', 'Age'],
                             prefix_sep='_',
                             dummy_na=False,
                             columns=['Geography', 'Gender', 'Age_Bin'],
                             sparse=False,
                             drop_first=False,
                             dtype=int)
Churn_model

plt.show()

visualize_svm()










