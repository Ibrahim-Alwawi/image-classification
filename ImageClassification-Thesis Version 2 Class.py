
# coding: utf-8

# In[1]:

import cv2         #import opencv
from matplotlib import pyplot as plt
import os


# In[2]:

train_images = {}     #container for training images
train_histograms = {}    #container for trainig image histograms
path = "C:/images/eyes/training/"    #path to folder where training images are stored
for f in os.listdir(path):      #for each file in the folder
    im = cv2.imread(path+f)     #read the image file
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  #convert image color format from BGR to RGB
    train_images[f] = im   #store training image
    hist = cv2.calcHist([im], [0], None, [8], [0,255]) #calculate histogram
    hist = cv2.normalize(hist).flatten()  #normalise histogram
    train_histograms[f] = hist  #store histogram 


# In[3]:

test_images = {}
test_histograms = {}
path = "C:/images/eyes/test/"
for f in os.listdir(path):
    im = cv2.imread(path+f)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    test_images[f] = im
    hist = cv2.calcHist([im], [0], None, [8], [0,255])
    hist = cv2.normalize(hist).flatten()
    test_histograms[f] = hist


# In[4]:

#OPENCV_METHODS = (("Correlation", cv2.cv.CV_COMP_CORREL),("Chi-Squared", cv2.cv.CV_COMP_CHISQR),("Intersection", cv2.cv.CV_COMP_INTERSECT)
#(Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA))


# In[5]:

img_results = {}  #container for saving results in memory
for (k1, hist1) in test_histograms.items():  #For each test histogram
    results = {}    
    for (k, hist) in train_histograms.items(): #For each training image histogram
        d = cv2.compareHist(hist1, hist, cv2.cv.CV_COMP_BHATTACHARYYA)  #calculate.... 
        #  distance between test image histogram and training image histogram
        results[k] = d;  #save distance
    results = sorted([(v,k) for (k,v) in results.items()], reverse=False) #sort results...
    # according to distance 
    img_results[k1] = results[0][1]  # get and save nearest neighbour


# In[6]:

labels = {'s':'Severe', 'n':'Normal'}
for i, (k,v) in enumerate(img_results.items()): #for each test image and nearest neighbour
    label = ''
    if v[:1].lower() in labels:
        label = labels[v[:1].lower()]
    print "Patient "+str(i+1)+": " + k + " => " + v + " (" +label+")\n"   #print test image and nearest neighbour


# In[ ]:




# 

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



