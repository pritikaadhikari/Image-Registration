# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:00:23 2023

@author: Pritika Adhikari
"""


import cv2
import numpy as np

def register_images(image1, image2,y):
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize the BRISK feature detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors for the two images
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Match keypoints between the two images using a brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2,None)
    
    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #cv2.imwrite('outputOUM/matchesOUM/OUM'+y+'.png',img_matches)

    # Extract the matched keypoints from the two images
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the homography matrix that maps the points from image1 to image2
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC)
    num_inliers = np.sum(mask)

    # Warp image1 to align with image2
    height, width, channels = image2.shape
    registered_image = cv2.warpPerspective(image1, homography, (width, height))
   

    return registered_image,matches,num_inliers






