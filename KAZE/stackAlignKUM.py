# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 14:19:29 2023

@author: Pritika Adhikari
"""


import cv2
import os
import numpy as np
from sewar.full_ref import scc, uqi, vifp, ssim, mse, psnr

os.chdir('C:/Users/dell/OneDrive/Desktop/MtechFinalProject/March2023/differentRegistrations/differentRegistrations/KAZE')

from registrationKUM import register_images

x = 1
images = list()
avg_match_len=0
mean_ssim=0
mean_mse=0
mean_psnr=0
mean_rmse=0
mean_ssc=0
mean_uqi=0
mean_vifp=0
inliers=0
while(True):
    
    y = str(x)
    img1_color = cv2.imread(r'../outputAugmented/test'+y+'.png')  # Image to be aligned.
    img2_color = cv2.imread(r'../outputAugmented/test0.png')    # Reference image.
    
    registered_img,no_of_matches,num_inliers = register_images(img1_color,img2_color,y) # Image registration
    #print(len(no_of_matches))
    avg_match_len=avg_match_len+len(no_of_matches)
    inliers = inliers+num_inliers
    #cv2.imwrite('outputKUM/regKUM/KUM'+y+'.png',registered_img)


    images.append(registered_img)
    
    #calcuate ssim
    grey2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    reg_gray=cv2.cvtColor(registered_img, cv2.COLOR_BGR2GRAY)
    ssim_value=ssim(grey2,reg_gray)
    ssim_value = ssim_value[0]
    mean_ssim=mean_ssim+ssim_value
    
    #calculate mse
    mse_value = mse(img2_color, registered_img)
    mean_mse=mean_mse+mse_value
    #calculate rmse
    rmse_value = np.sqrt(mse_value)
    mean_rmse=mean_rmse+rmse_value
    
    #calculate psnr
    psnr_value = psnr(img2_color, registered_img)
    mean_psnr=mean_psnr+psnr_value
    
    #claculate ssc
    ssc_value=scc(img2_color,registered_img)
    mean_ssc=mean_ssc+ssc_value
    
    uqi_value=uqi(img2_color,registered_img)
    mean_uqi=mean_uqi+uqi_value
    
    vifp_value=vifp(img2_color,registered_img)
    mean_vifp=mean_vifp+vifp_value
    
    
    
    x =x+1
    
    if(x>100):
        break
    
precision= (inliers/100)/(avg_match_len/100)
 
print("avg match no",int(avg_match_len/100))
print('avg ssim:',float(mean_ssim/100))
print("avg mse:",float(mean_mse/100))
print("avg psnr:",float(mean_psnr/100))
print("avg rmse:",float(mean_rmse/100))
print("avg scc:",float(mean_ssc/100))
print("avg uqi:",float(mean_uqi/100))
print("avg vfp:",float(mean_vifp/100))
print("no. of inliers:",int(inliers/100))
print("precision:",float(precision))
