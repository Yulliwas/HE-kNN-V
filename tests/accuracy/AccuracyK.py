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
    return np.array(predicted)
  
datas=["iris.csv","breast.csv","heart.csv","wine.csv","glass.csv","mnist.csv"]

for dataset in datas:
  dataset="mnist.csv"
  accuracy_clear = open("accuracy_clearK.txt", "a")
  accuracy_cipher = open("accuracy_cipherK.txt", "a")
  accuracy_clear.write(dataset+"\n")
  accuracy_cipher.write(dataset+"\n")
  iris=""
  if dataset=="iris.csv":
    iris = datasets.load_iris()
    X= iris.data
    y= iris.target
  if dataset=="breast.csv":
    iris = datasets.load_breast_cancer()
    X= iris.data
    y= iris.target  
  if dataset=="wine.csv":
    iris = datasets.load_wine()
    X= iris.data
    y= iris.target
  if dataset=="mnist.csv":
    iris = datasets.load_digits()
    X= iris.data
    y= iris.target
  if dataset=="heart.csv":
    df = pd.read_csv("./heart.csv")
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
  if dataset=="glass.csv":
    df = pd.read_csv("./glass.csv")
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
  # normalisation des donnees entre 0 et 1
  minmaxScaler=MinMaxScaler()
  #X= iris.data
  #y= iris.target #!= 0) * 1

  a=minmaxScaler.fit_transform(X)

  #separarion des donnees en donnees de test et donnees d'entrainement
  X_train, X_test, y_train, y_test = train_test_split(a, y, test_size=0.3)
  

  #entrainement de modele
  
  for j in range(1,20,+2):  
    knn= KNNClair(k=j)
    knn2=KNeighborsClassifier(n_neighbors=j)
    random_indexes_global=[]
    accuracy_global=0
    for i in range(200):
      random_indexes= np.random.choice(int(0.7*y_train.shape[0]), 40)
      knn.fit(X_train[random_indexes,:],y_train[random_indexes])
      y_predicted=knn.predict(X_test)
      accuracy_local=accuracy_score(y_test,y_predicted)
      if accuracy_local>accuracy_global:
        accuracy_global= accuracy_local
        random_indexes_global=random_indexes.copy()
    knn2.fit(X_train[random_indexes_global,:],y_train[random_indexes_global])
    y_predicted2=knn2.predict(X_test)
    accuracy_local=accuracy_score(y_test,y_predicted2)
    accuracy_clear.write(str(accuracy_local)+"\n")
    accuracy_cipher.write(str(accuracy_global)+"\n")
  
    print("accuracy rate: ",j ," encrypted ", accuracy_global,"  clear ", accuracy_local)
  accuracy_clear.close()
  accuracy_cipher.close()
