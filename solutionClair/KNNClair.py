import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import accuracy_score, recall_score
from sklearn.cluster import KMeans
from sklearn import datasets
from scipy.spatial import distance_matrix
from scipy import stats

class KNNClair:
  
  def __init__(self,k=3):
    self.k = k
  
  def fit(self,X_train,y_train):
    self.X_train = X_train
    self.y_train = y_train
  
  def formule2(self,x_train1,x_train2,x_test):
    x1_norm = np.linalg.norm(x_train1)/4.
    x2_norm = np.linalg.norm(x_train2)/4.
    Cj_Ci=np.around(100.*(x_train2-x_train1)/4.)
    Ai_Aj = x1_norm**2 - x2_norm**2
    
    return (Ai_Aj+2*(np.dot(Cj_Ci.T,x_test/400.)))>0
  
  def construire_delta_matrix(self,x):
    #dists=np.sqrt(np.sum(np.square(self.X_train-x),axis=1)).reshape((self.X_train.shape[0],1))
    delta_matrix = np.empty((self.X_train.shape[0],self.X_train.shape[0]))
    for i in range(self.X_train.shape[0]):
      for j in range(self.X_train.shape[0]):
        delta_matrix[i,j]=self.formule2(self.X_train[i,:],self.X_train[j,:],x)#(dists[i]-dists[j])>0
    return delta_matrix
  
  def scoring_operation(self, delta_matrix):
    return delta_matrix.sum(axis=0)
  
  def predict_single(self, x):
    delta_matrix= self.construire_delta_matrix(x)
    scores= self.scoring_operation(delta_matrix)
    args=np.argsort(scores)[-self.k:]
    return stats.mode(self.y_train[args])[0]

  def predict(self,X_test):
    predicted=[]
    for x in X_test:
      predicted.append(self.predict_single(x))
    return np.array(predicte
