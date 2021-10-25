import numpy as np

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

#Keys of cancer_dataset
print("Keys of cancer dataset: {}".format(cancer.keys()) + "\n")

#Information regarding Target in dataset
print("Target names: {}".format(cancer.target_names) + "\n")

print("Type of target: {}".format(type(cancer.target)) + "\n")

print("Shape of target: {}".format(type(cancer.target.shape)) + "\n")

print("Target: {}".format((cancer.target)) + "\n")

#Shape of data
print("Shape of cancer data: {}".format(cancer.data.shape) + "\n")

#Frequency of classes
print("Sample counts per class: {}".format({n:v for n,v in zip(cancer.target_names, np.bincount(cancer.target))}) + "\n")

#Features in dataset
print("Feature names:\n{}".format(cancer.feature_names) + "\n")

#Generelized information regarding cancer_dataset
print(cancer['DESCR'][:3000] + "\n")
