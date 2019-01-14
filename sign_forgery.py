import numpy as mynp
 
import cv2
 
from matplotlib import pyplot as plt
 
MIN_COUNT = 10
 
myimage1 = cv2.imread('ChequewithCopiedSignature.png',0)
 
myimage2 = cv2.imread('OriginalCheque.png',0)
 
sift = cv2.xfeatures2d.SIFT_create()

 
# find the keypoints and descriptors with SIFT
 
k1, im1 = sift.detectAndCompute(myimage1,None)
 
k2, im2 = sift.detectAndCompute(myimage2,None)
 
FLANN_INDEX_KDTREE = 0
 
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
 
search_params = dict(checks = 50)
 
flann = cv2.FlannBasedMatcher(index_params, search_params)
 
matches = flann.knnMatch(im1,im2,k=2)

 
# store all the good matches as per Lowe’s ratio test.
 
good = []
 
for m,n in matches:
 
if m.distance < 0.7*n.distance:
 
good.append(m)
 
if len(good)>MIN_COUNT:
 
src_pts = mynp.float32([ k1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
 
dst_pts = mynp.float32([ k2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
 
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
 
matchesMask = mask.ravel().tolist()
 
h,w = myimage1.shape
 
pts = mynp.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
 
dst = cv2.perspectiveTransform(pts,M)
 
myimage2 = cv2.polylines(myimage2,[mynp.int32(dst)],True,255,3, cv2.LINE_AA)
 
else:
 
print "Not enough matches are found - %d/%d" % (len(good),MIN_COUNT)
 
matchesMask = None
 
draw_params = dict(matchColor = (0,255,0), /*draw matches in green color*/ singlePointColor = None,matchesMask = matchesMask, /*draw only inliers*/
 
flags = 2)
 
myimg9 = cv2.drawMatches(myimage1,k1,myimage2,k2,good,None,**draw_params)
 
plt.imshow(myimg9, 'gray'),plt.show()
