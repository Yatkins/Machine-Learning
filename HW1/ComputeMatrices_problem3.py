#Problmen 3

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets

# first function to fill, compute distance matrix using loops
def compute_distance_naive(X):
    N = X.shape[0]      # num of rows
    D = X[0].shape[0]   # num of cols
    
    M = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            xi = X[i,:]
            xj = X[j,:]
            #dist = 0.0  #a placetaker line, 
                        # you have to change it to distance between xi and xj
            dist = sum([(xi[i]-xj[i])**2 for i in range(len(xi))])**.5
            M[i,j] = dist

    return M

# second function to fill, compute distance matrix without loops
def compute_distance_smart(X):
    N = X.shape[0]  # num of rows
    D = X[0].shape[0]  # num of cols
    
    # use X to create M
    M = np.zeros([N, N])

    M = np.sum((X[None,:] - X[:, None])**2, -1)**0.5
    
    return M

# third function to fill, compute correlation matrix using loops
def compute_correlation_naive(X):
    N = X.shape[0]  # num of rows
    D = X[0].shape[0]  # num of cols

    # use X to create M
    M = np.zeros([D, D])

    M = np.zeros([D, D])
    for i in range(D):
        for j in range(D):
            xi = X[:,i] - np.average(X[:,i])
            xj = X[:,j] - np.average(X[:,j])
            corr = np.divide(np.dot(xi,xj), np.dot(np.sqrt(np.dot(xi,xi)), np.sqrt(np.dot(xj,xj))))
            M[i, j] = corr

    return M

# fourth function to fill, compute correlation matrix without loops
def compute_correlation_smart(X):
    N = X.shape[0]  # num of rows
    D = X[0].shape[0]  # num of cols

    # use X to create M
    M = np.zeros([D, D])
    
    P = X - X.mean(axis=0)
    h,w = (P).shape
    cov = (P.T @ P)/(h-1)
    dinv = np.diag(1 / np.sqrt(np.diag(cov))) 
    M = dinv @ cov @ dinv

    return M

def main():
    
    iris_data = np.asarray(datasets.load_iris().data)
    breast_cancer_data = np.asarray(datasets.load_breast_cancer().data)
    digits_data = np.asarray(datasets.load_digits().data)

    data_sets = [iris_data, breast_cancer_data, digits_data]

    distance_times = np.zeros([3, 2])
    correlation_times = np.zeros([3, 2])

    for i in range(3):
        X = data_sets[i]
        st = time.time()
        dist_loop = compute_distance_naive(X)
        et = time.time()
        distance_times[i,0] = et-st
        
        st = time.time()
        dist_loop = compute_distance_smart(X)
        et = time.time()
        distance_times[i,1] = et-st
        
        st = time.time()
        dist_loop = compute_correlation_naive(X)
        et = time.time()
        correlation_times[i,0] = et-st        
        
        st = time.time()
        dist_loop = compute_correlation_smart(X)
        et = time.time()
        correlation_times[i,1] = et-st
        
    distance_t = pd.DataFrame(distance_times, columns = ['Naive', 'Smart'], index = ['Iris', 'Breast Cancer', 'Digits'])
        
    print(distance_t)
    
    correlation_t = pd.DataFrame(correlation_times, columns = ['Naive', 'Smart'], index = ['Iris', 'Breast Cancer', 'Digits'])
        
    print(correlation_t)
        
if __name__ == "__main__": main()