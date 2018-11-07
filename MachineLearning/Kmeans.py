import numpy as np
import cv2
from matplotlib import pyplot as plt

# Feature set containing (x,y) values of xxx known/training data
trainSet = np.random.randint(20,80,(1000,2)).astype(np.float32)

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0 )

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS

# kmeans(data, K, bestLabels, criteria, attempts, flags[, centers]) -> retval, bestLabels, centers
compactness, labels, centers = cv2.kmeans(trainSet, 3, None, criteria, 10, flags)

# Now separate the data, Note the flatten()
L0 = trainSet[labels.ravel()==0]
L1 = trainSet[labels.ravel()==1]
L2 = trainSet[labels.ravel()==2]

# Plot the data
plt.scatter(L0[:,0],L0[:,1],20,'b')
plt.scatter(L1[:,0],L1[:,1],20,'r')
plt.scatter(L2[:,0],L2[:,1],20,'k')
plt.scatter(centers[:,0],centers[:,1],80,'y','s')
plt.show()
