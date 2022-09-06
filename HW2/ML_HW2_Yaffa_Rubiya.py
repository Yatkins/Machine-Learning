#import packages
import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load the breast cancer data
breast_cancer = sklearn.datasets.load_breast_cancer()
X = breast_cancer.data
Y= breast_cancer.target


class Perceptron:
  
  #constructor
  def __init__ (self, X_train, Y_train, X_test, Y_test):
    self.w = None
    self.b = None
    self.X = X_train
    self.Y = Y_train
    self.X_test = X_test
    self.Y_test = Y_test
    
  #model  
  def model(self, x):
    
    return 1 if (np.dot(self.w, x) >= self.b) else 0
  
  #predictor to predict on the data based on w
  def predict(self, X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
    return np.array(Y)
    
  def fit(self, epochs, lr, pseudo_inv=False):
   
    if pseudo_inv:
          
          x_pseudo = np.linalg.pinv(self.X)
          self.w = np.dot(x_pseudo,self.Y)
    else:
     self.w = np.ones(X.shape[1])
     
    errors = {}
    max_errors = 0
    self.b = 0
    wt_matrix = []
     #for all epochs 
    for i in range(epochs):
       for x, y in zip(X, Y):
         y_pred = self.model(x)
         if y == 1 and y_pred == 0:
           self.w = self.w + lr * x
           self.b = self.b - lr * 1
         elif y == 0 and y_pred == 1:
           self.w = self.w - lr * x
           self.b = self.b + lr * 1
          
       wt_matrix.append(self.w)    
       errors[i] = (1-accuracy_score(self.predict(X_test), Y_test))
       if (errors[i] > max_errors):
           max_errors = errors[i]
           chkptw = self.w
           chkptb = self.b
    #checkpoint (Save the weights and b value)
    self.w = chkptw
    self.b = chkptb
    
    #print("The max accuracy of the model with training data", max_accuracy)
    #plot the accuracy values over epochs
    return errors 
     
    
  def plot_accuracy(self, model1, model2, test_size):
        
        
        
        
        plt.figure(figsize=(10,5))
        plt.plot(model1.values(), 'r',label= "data points")
        plt.plot(model2.values(), 'g',label= "pseudo solution for training")
        plt.xlabel("# of iterations")
        plt.ylabel("errors out sample (Eout)")
        plt.title(test_size)
        plt.ylim([0,0.7])
        #plt.xlim([200,500])
        plt.legend()
        plt.title("test size = " + str(test_size))
        plt.show()
    
test_sizes = [0.1,0.2,0.3,0.4,0.5]
#5 different Ds
for test_size in test_sizes:
  
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = test_size, stratify = Y, random_state = 1)
    p = Perceptron(X_train, Y_train, X_test,Y_test)
    model_1 = p.fit(500, 0.01, pseudo_inv = 0)
    model_2 = p.fit(500, 0.01, pseudo_inv=1)
    p.plot_accuracy(model_1, model_2, test_size )