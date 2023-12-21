import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df=pd.read_csv('/content/sample_data/california_housing_train.csv')

class LinearRegression():
  def __init__(self, learning_rate, iters):
    self.iters=iters
    self.learning_rate=learning_rate

  def fit(self,X,Y):
    self.m, self.n= X.shape
    self.X=X
    self.Y=Y
    self.b=0
    self.w=np.zeros(self.n)

    for i in range(self.iters):
      self.update_weights()

  def update_weights(self):
    Y_pred=self.predict(self.X)
    dw=-2*(self.X.T).dot(self.Y-Y_pred)/self.m
    db=-2*np.sum(self.Y-Y_pred)/self.m
    self.w= self.w-self.learning_rate*dw
    self.b= self.b-self.learning_rate*db

  def predict(self, X):
    return X.dot(self.w)+self.b


def main():
  X=df[['latitude','housing_median_age','median_income']]
  Y=df['median_house_value']

  X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=1/3, random_state=0)
  #random_state=None

  model=LinearRegression(learning_rate=0.0002, iters=1000)
  # model = LinearRegression()
  # model._init_(0.0002,1000)

  model.fit(X_train, Y_train)

  Y_pred=model.predict(X_test)

  print(X)
  print("Predicted values: ",Y_pred[:10])
  print("Actual values: ", Y_test[:10])

  print("Weight: ", model.w)
  print("Bias: ", model.b)

if __name__=="__main__":
  main()
