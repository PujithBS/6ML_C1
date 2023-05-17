import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\STUDENT\Desktop\salary_Data.csv")
df.head()
df.shape
X = df['YearsExperience'].values.reshape(-1, 1)
Y = df['Salary'].values.reshape(-1, 1)
class Linear_Regression():

	def __init__(self, learning_rate, no_of_itr):
		self.learning_rate = learning_rate
		self.no_of_itr = no_of_itr

	def fit(self, X, Y):


		self.m, self.n = X.shape	 # Number of rows and columns
		# Initiating the weight and bias
		self.w = np.zeros((self.n, 1))
		self.b = 0
		self.X = X
		self.Y = Y

		# Implementing the gradient descent.
		for i in range(self.no_of_itr):
			self.update_weigths()

	def update_weigths(self):
		Y_prediction = self.predict(self.X)

		# Calculating gradients
		dw = -(self.X.T).dot(self.Y - Y_prediction)/self.m

		db = -np.sum(self.Y - Y_prediction)/self.m

		# Updating weights
		self.w = self.w - self.learning_rate * dw
		self.b = self.b - self.learning_rate * db

	def predict(self, X):
		return X.dot(self.w) + self.b

	def print_weights(self):
		print('Weights for the respective features are :')
		print(self.w)
		print()

		print('Bias value for the regression is ', self.b)
model = Linear_Regression(learning_rate=0.03,no_of_itr=2000)
model.fit(X,Y)
model.print_weights()
plt.scatter(df['YearsExperience'], df['Salary'])
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Exp v/s Salary')

X = df['YearsExperience'].values
plt.plot(X, 9988 * X + 23876)
plt.show()
