import numpy as np
from Q1a import  *

def knn_model(x_train, y, x_test, k, metric='euclidean'):
    """
    Finds the k nearest neighbors of xTest in xTrain.
    Input:
    xTrain = n x d matrix. n=rows and d=features
    y = x_train labels
    xTest = m x d matrix. m=rows and d=features (same amount of features as xTrain)
    k = number of nearest neighbors to be found
    metric: the distance metric to be used (euclidean, manhattan, minkowski)
    Output:
    dists = distances between xTrain/xTest points. Size of n x m
    indices = kxm matrix with indices of yTrain labels
    """

    if metric == 'euclidean':
        #the following formula calculates the Euclidean distances.
        # distances = -2 * x_train @ x_test.T + np.sum(x_test**2,axis=1) + np.sum(x_train**2,axis=1)[:, np.newaxis]
        distances = euclidean_distance(x=x_train, y=x_test)
        distances[distances < 0] = 0
        distances = distances**.5
    if metric == 'manhattan':
        distances = manhattan_distance(x_train, x_test)

    if metric == 'minkowski':
        distances = minkowski_distance(x_train, x_test)
 
    #returning the top-k closest distances.
    indices = np.argsort(distances, 0)[0:k, : ]#get indices of sorted items
    distances = np.sort(distances,0)[0:k, : ] #distances sorted in axis 0

    yTrain = y.flatten()
    rows, columns = indices.shape
    predictions = list()
    for j in range(columns):
        temp = list()
        for i in range(rows):
            cell = indices[i][j]
            temp.append(yTrain[cell])
        predictions.append(max(temp,key=temp.count)) #this is the key function, brings the mode value
    predictions=np.array(predictions)
    return predictions

if __name__ == '__main__':
    X, y = load_data('data1.txt')
    test = load_test('test.txt')
    print(knn_model(X, y, test, 4, 'euclidean'))
    # print(knn_predictions(X, y, test, 5))
