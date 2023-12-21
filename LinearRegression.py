import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

class LinearRegression():
    def _init_(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, Y):
        print(X.shape)
        self.m, self.n = X.shape
        print(self.m)
        print(self.n)
        self.W = np.zeros([1,self.n])
        # self.W = 0
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
    df = pd.read_csv("/content/sample_data/Housing.csv")

    X = df['lotsize']
    Y = df['price']

    # Splitting dataset into train and test set
    X=X.values.reshape(len(X),1)
    Y=Y.values.reshape(len(Y),1)

    # Split the data into training/testing sets
    X_train = X[:-250]
    X_test = X[-250:]

    # Split the targets into training/testing sets
    Y_train = Y[:-250]
    Y_test = Y[-250:]
    print(X_train.shape)

    # Model training
    model = LinearRegression(iterations=1000, learning_rate=0.0000000002)
    model.fit(X_train, Y_train)
    # Prediction on test set
    Y_pred = model.predict(X_test)
    print(Y_pred.shape)
    # print(Y_pred)
    # Print predictions and actual values
    print("Predicted values: ", np.round(Y_pred[:100], 2))
    print("Actual values:    ", Y_test[:100])

    # Print trained weights (slope) and bias
    print(model.W.shape)
    print(model.X.shape)
    print(model.Y.shape)
    print("Trained slope (W): ", model.W[0])
    print("Trained bias (b):  ", model.b)
    # print(sklearn.metrics.r2_score(X_test, Y_pred))
    # Visualization on test set
    # plt.scatter(X_test, Y_test, color='blue')
    # plt.plot(X_test, Y_pred, color='orange')
    # plt.title('Test data')
    # plt.xlabel('Lotsize')
    # plt.ylabel('Price')
    # plt.show()

if _name_ == "_main_":
    main()
