import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()

        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.m
        db = - 2 * np.sum(self.Y - Y_pred) / self.m
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        return X.dot(self.W) + self.b

def main():
    # Importing dataset
    df = pd.read_csv("/content/sample_data/california_housing_train.csv")

    X = df[['latitude','housing_median_age','median_income']]

    Y = df['median_house_value']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0 )

    # Model training
    model = LinearRegression(iterations=1000, learning_rate=0.00002)
    # print(X.shape)
    model.fit(X_train, Y_train)

    # Prediction on test set
    Y_pred = model.predict(X_test)

    # Print predictions and actual values
    print("Predicted values: ", np.round(Y_pred[:10], 2))
    print("Actual values:    ", Y_test[:10])
    print(X)

    # Print trained weights (slope) and bias
    print("Trained slope (W): ", np.round(model.W, 2))
    print("Trained bias (b):  ", np.round(model.b, 2))

if __name__ == "__main__":
    main()