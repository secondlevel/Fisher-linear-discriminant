#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis


def accuracy(pred, target):  # Calculate accuracy score
    correct = (pred == target)
    return np.sum(correct)/len(target)


# ## HW2: Linear Discriminant Analysis
# In hw2, you need to implement Fisher’s linear discriminant by using only numpy, then train your implemented model by the provided dataset and test the performance with testing data
#
# Please note that only **NUMPY** can be used to implement your model, you will get no points by simply calling sklearn.discriminant_analysis.LinearDiscriminantAnalysis

y_pred = []

# ## Load data

x_train = pd.read_csv("x_train.csv").values
y_train = pd.read_csv("y_train.csv").values[:, 0]
x_test = pd.read_csv("x_test.csv").values
y_test = pd.read_csv("y_test.csv").values[:, 0]

# ## 1. Compute the mean vectors mi, (i=1,2) of each 2 classes

# Your code HERE

# Get the training data in different class(0->first, 1->second)
x1 = x_train[y_train == 0]
x2 = x_train[y_train == 1]

# Calculate the average point in different cluster(x1->first, x2->second)
m1 = np.mean(x1, axis=0)
m2 = np.mean(x2, axis=0)

assert m1.shape == (2,)
assert m2.shape == (2,)
print(f"mean vector of class 1: {m1}\n"f"mean vector of class 2: {m2}")


# ## 2. Compute the Within-class scatter matrix SW

# Your code HERE

# Calculate within-class covariance matrix(sw) through data point and average point.
sw = ((x1-m1).T @ (x1-m1)) + ((x2-m2).T @ (x2-m2))

assert sw.shape == (2, 2)
print(f"Within-class scatter matrix SW: {sw}")


# ## 3.  Compute the Between-class scatter matrix SB

# Your code HERE

sb = ((m2-m1).reshape(-1, 1)) @ ((m2-m1).reshape(-1, 1).T)


assert sb.shape == (2, 2)
print(f"Between-class scatter matrix SB: {sb}")

# ## 4. Compute the Fisher’s linear discriminant

# Your code HERE

# Calculate w through sw, cluster 1 mean, cluster 2 mean
w = (np.linalg.inv(sw) @ (m2-m1)).reshape(-1, 1)

assert w.shape == (2, 1)
print(f"Fisher’s linear discriminant: {w}")

# ## 5. Project the test data by linear discriminant to get the class prediction by nearest-neighbor rule and calculate the accuracy score
# you can use accuracy_score function from sklearn.metric.accuracy_score

# setting the value of k
k = 3

# project data point to project data
train_project = (x_train @ w) * (w/np.sum(w*w)).reshape(-1,)
test_project = (x_test @ w) * (w/np.sum(w*w)).reshape(-1,)

# find the k-nearst, and the k-nearst training label which occurs most times would be the predicted label.
for test_point in test_project:

    distance_list = []

    for train_point in train_project:
        distance = np.sum((test_point-train_point)**2)
        distance_list.append(distance)

    distance_list = np.array(distance_list).reshape(-1,)
    index = np.argpartition(distance_list, k)[:k]
    exact_value = y_train[index]
    pred_value = np.argmax(np.bincount(exact_value))
    y_pred.append(pred_value)

y_pred = np.array(y_pred)

acc = accuracy(y_pred, y_test)

print(f"Accuracy of test-set {acc}")

# ## 6. Plot the 1) best projection line on the training data and show the slope and intercept on the title (you can choose any value of intercept for better visualization) 2) colorize the data with each class 3) project all data points on your projection line. Your result should look like [this image](https://i.imgur.com/tubMQpw.jpg)

slope, intercept = np.polyfit(train_project[:, 0], train_project[:, 1], 1)

# original testing data
train_x1 = x_train[y_train == 0]
train_x2 = x_train[y_train == 1]

# project testing data
train_x3 = train_project[y_train == 0]
train_x4 = train_project[y_train == 1]

# define the figure size
plt.figure(figsize=(10, 10))

plt.title('Projection Line: w=%f, b=%f' % (slope, intercept), fontsize=17)

# show the red class data point
plt.scatter(train_x1[:, 0], train_x1[:, 1], s=40, c='red')
plt.scatter(train_x2[:, 0], train_x2[:, 1], s=40, c='blue')

# show the blue class data point
plt.scatter(train_x3[:, 0], train_x3[:, 1], s=30, c='red')
plt.scatter(train_x4[:, 0], train_x4[:, 1], s=30, c='blue')

# show the project line y=ax+b
plt.plot(train_project[:, 0], train_project[:, 1], alpha=0.5, color="blue")

# show the red class data point to project point
for point in np.c_[train_x1, train_x3]:
    plt.plot([point[0], point[2]], [point[1], point[3]],
             alpha=0.1, color="red")

# show the blue class data point to project point
for point in np.c_[train_x2, train_x4]:
    plt.plot([point[0], point[2]], [point[1], point[3]],
             alpha=0.1, color="blue")

plt.show()
