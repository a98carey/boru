import cv2 
import numpy as np

img = cv2.imread("cross.jpg",0)

h = img.shape[0]
w = img.shape[1]

Mask_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
Mask_y = [[1,2,1],[0,0,0],[-1,-2,-1]]

Sobel_x = np.zeros((h, w, 1),dtype = np.uint8)
Sobel_y = np.zeros((h, w, 1),dtype = np.uint8) 

sum_x = 0
sum_y = 0

for x in range(1,h-1):
	for y in range(1,w-1):
		sum_x = 0
		sum_y = 0
		for a in range(3):
			for b in range(3):
				xn = x + a -1
				yn = y + b -1
				
				# convolution
				sum_x = sum_x + img[xn][yn] * Mask_x[a][b] 
				sum_y = sum_y + img[xn][yn] * Mask_y[a][b]

		# /4:正規化  4: 強度 1+2+1
		Sobel_x[x][y] = abs(sum_x/4) 
		Sobel_y[x][y] = abs(sum_y/4) 

Sobel = abs(Sobel_x) + abs(Sobel_y)

print(img.shape)

cv2.imshow("Sobel_x",Sobel_x)
cv2.imshow("Sobel_y",Sobel_y)
cv2.imshow("Sobel",Sobel)

cv2.waitKey(0)  
cv2.destroyAllWindows()