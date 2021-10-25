from scipy.spatial import distance

def euc(a,b):
	return distance.euclidean(a,b)

class CustomKNN():
	def fit(self, trainingdata, trainingtarget):
		self.trainingdata = trainingdata
		self.trainingtarget = trainingtarget

	def predict(self, testdata):
		predictions = []
		for row in testdata:
			label = self.closest(row)
			predictions.append(label)
		return predictions

	def closest(self, row):
		best_dist = euc(row, self.trainingdata[0])
		best_index = 0
		for i in range(1, len(self.trainingdata)):
			dist = euc(row, self.trainingdata[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.trainingtarget[best_index]