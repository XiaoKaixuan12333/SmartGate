from sklearn import metrics
from sklearn.metrics import davies_bouldin_score


def calculate_hybrid_index(data,labels):
    #data is the matrix of samples and features
    #labels the all the result of cluster

    #output is the Hybrid index value
    sc=metrics.silhouette_score(data, labels, metric='euclidean')
    ch=metrics.calinski_harabasz_score(data, labels)
    db=davies_bouldin_score(data, labels)
    hybrid=sc+ch-db
    return hybrid