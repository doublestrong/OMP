import numpy as np
from sklearn.cluster import KMeans

def  kMeansDictionary(training, k):
    est = KMeans(n_clusters=k,init='k-means++',tol=0.0001,verbose=1).fit(training)
    return est

def VLAD(X,visualDictionary):

    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    k=visualDictionary.n_clusters
   
    m,d = X.shape
    V=np.zeros([k,d])

    for i in range(k):
        if np.sum(predictedLabels==i)>0:
            V[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)
    

    V = V.flatten()
    V = np.sign(V)*np.sqrt(np.abs(V))
    V = V/np.sqrt(np.dot(V,V))
    return V






	




