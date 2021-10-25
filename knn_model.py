import numpy as np
import matplotlib.pyplot as plt
import KNearestNeighbor
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def main():
	model()

def model():
	cancer = load_breast_cancer()
	train_data, test_data, train_target, test_target = train_test_split(cancer.data, cancer.target, test_size = 0.3)

	classifier = KNearestNeighbor.CustomKNN()
	classifier.fit(train_data, train_target)
	prediction = classifier.predict(test_data)
	
	accuracy = accuracy_score(test_target, prediction)
	print("Accuracy: ", accuracy*100)

if __name__ == "__main__":
	main()
