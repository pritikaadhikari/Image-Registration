# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 18:54:28 2023

@author: Pritika Adhikari
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
os.chdir('C:/Users/dell/OneDrive/Desktop/MtechFinalProject/March2023/differentRegistrations/differentRegistrations')
df=pd.read_excel("evalMet.xlsx",sheet_name=0) #sheet_nmae=0 for eval sheet

'''
x1 = ["OR", "OUA", "OUM"]
x2=["AR", "AUA","AUM"]
x3=["KR","KUA","KUM"]
x4=["BR","BUA","BUM"]
'''
cats=["OR", "OUA", "OUM","AR", "AUA","AUM","KR","KUA","KUM","BR","BUA","BUM"]
for i in df.columns:
    
    print(i)
    y1=df.loc[[0,1,2], [i]]
    y2=df.loc[[3, 4, 5], [i]]
    y3=df.loc[[6, 7, 8], [i]]
    y4=df.loc[[9,10, 11], [i]]
    plt.bar(x1,y1, color="blue", label="ORB")
    plt.bar(x2,y2, color="magenta", label="AKAZE")
    plt.bar(x3,y3, color="green", label="KAZE")
    plt.bar(x4,y4, color="saddlebrown", label="BRISK")
    plt.plot(x1, y1, color='blue' )
    plt.plot(x2,y2, color="magenta")
    plt.plot(x3,y3, color="green")
    plt.plot(x4,y4, color="saddlebrown")
    
    plt.legend()

    # Set title and axis labels
    plt.title("{} of Different Methods\n".format(i))
    plt.xlabel("Methods")
    plt.ylabel(i)

    # Display the chart
    plt.show()
    
"""
#time plot
x1 = ["OR", "OUA", "OUM"]
x2=["AR", "AUA","AUM"]
x3=["KR","KUA","KUM"]
x4=["BR","BUA","BUM"]
y1=[29.117335800023284,29.334949299984146, 37.75934089999646]
y2=[38.53148720000172,28.74440130003495,33.51908369996818]
y3=[32.86123000003863,32.66479309997521,32.44154410000192]
y4=[30.526041400036775,36.374440899991896,39.931214999989606]
plt.scatter(x1,y1, color="blue", marker="o", label="ORB")
plt.scatter(x2,y2, color="magenta", marker="s", label="AKAZE")
plt.scatter(x3,y3, color="green", marker="^", label="KAZE")
plt.scatter(x4,y4, color="saddlebrown", marker="d", label="BRISK")
plt.plot(x1, y1, color='blue', linestyle='-', linewidth=1)
plt.plot(x2,y2, color="magenta", linestyle='-', linewidth=1)
plt.plot(x3,y3, color="green", linestyle='-', linewidth=1)
plt.plot(x4,y4, color="saddlebrown", linestyle='-', linewidth=1)

plt.legend()

# Set title and axis labels
plt.title("Runtime of Different Methods\n")
plt.xlabel("Methods")
plt.ylabel("Runtime")

# Display the chart
plt.show()

"""
