# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 14:17:42 2023

@author: Pritika Adhikari
"""


import cv2
import numpy as np

def register_images(image1, image2,y):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Detect KAZE features and compute descriptors for both images
    kaze = cv2.KAZE_create()
    keypoints1, descriptors1 = kaze.detectAndCompute(img1, None)
    keypoints2, descriptors2 = kaze.detectAndCompute(img2, None)
    
    # Use brute-force matching to find matching descriptors between the two images
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    
    # Sort the matches by their distance (lower is better) and keep only the top y matches
    matches = sorted(matches, key=lambda x:x.distance)
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #cv2.imwrite('outputKR/matchesKR/KR'+y+'.png',img_matches)
    
    # Get the coordinates of the matched keypoints in both images
    pts1 = []
    pts2 = []
    for match in matches:
        pts1.append(keypoints1[match.queryIdx].pt)
        pts2.append(keypoints2[match.trainIdx].pt)
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    
    # Find the homography matrix that maps pts1 to pts2
    homography,mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    num_inliers = np.sum(mask)
    
    # Use the homography matrix to warp img1 to the same perspective as img2
    height, width,channel = image2.shape
    registered_img = cv2.warpPerspective(image1, homography, (width, height))
    
    return registered_img, matches,num_inliers
