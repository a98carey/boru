import numpy as np
import cv2
import os 

def KNN(K, trainSet, trainLabel, testSet):

    knn = cv2.ml.KNearest_create()
    knn.train(trainSet, cv2.ml.ROW_SAMPLE, trainLabel)
    ret, results, neighbours, dist = knn.findNearest(testSet, k=K)

    return results

def Kmeans(clusters, trainSet):
    '''
    Kmeans(clusters, trainSet) -> labels, centers
    
    均值聚類演算法.

    @param clusters : 集群數
    @param trainSet : 訓練集

    @return labels  : 訓練結果標籤
    @return centers : 集群中心
    '''
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0 )

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    # kmeans(data, K, bestLabels, criteria, attempts, flags[, centers]) -> retval, bestLabels, centers
    compactness, labels, centers = cv2.kmeans(trainSet.astype('float32'), clusters, None, criteria, 10, flags)

    return labels, centers

if __name__ == '__main__':
    # read image
    filename = './digits.png'  
    src = cv2.imread(filename, 0)

    trainList = []
    testList  = []
    trainLabelList = []

    # split image
    cells = [np.hsplit(row, 100) for row in np.vsplit(src, 50)]

    trainList = [ i[:50] for i in cells ]
    testList  = [ i[50:] for i in cells ]

    trainSet = np.array(trainList, np.float32)
    testSet  = np.array(testList,  np.float32)
    trainSet = trainSet.reshape(2500, 400) 
    testSet  = testSet.reshape(2500,  400) 

    testSet20x20  = testSet.reshape(2500, 20, 20)

    # produce label
    for i in range(10):
        for j in range(250):
            trainLabelList.append(i) 
    trainLabel = np.array(trainLabelList, np.float32)   

    # KNN(K, trainSet, trainLabel, testSet) supervise
    results = KNN(3, trainSet, trainLabel, testSet)

    # Kmeans(clusters, trainSet) unsupervise
    # labels, centers = Kmeans(10, trainSet)

    # write image
    # for i in range(10):
    #     folder_path = './KNN/' + str(i) +'/'
    #     if os.path.exists(folder_path) == False:  #判斷資料夾是否存在  
    #         os.makedirs(folder_path)  # 創資料夾
    #         pass

    #     for j in range(2500):
    #         if int(results[j][0]) == i:
    #             save_path = folder_path + '/' + str(j)+'.jpg'
    #             cv2.imwrite(save_path, testSet20x20[j])
