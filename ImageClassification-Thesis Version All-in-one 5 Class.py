
# coding: utf-8

# In[36]:

import cv2         #import opencv
from matplotlib import pyplot as plt
import os


# In[40]:

path = "C:/images/experiments/"
for folder in os.listdir(path):
    train_images = {}     #container for training images
    train_histograms = {}    #container for trainig image histograms
    train_path = path+folder+"/train/"    #path to folder where training images are stored
    for f in os.listdir(train_path):      #for each file in the folder
        im = cv2.imread(train_path+f)     #read the image file
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  #convert image color format from BGR to RGB
        train_images[f] = im   #store training image
        hist = cv2.calcHist([im], [0], None, [8], [0,255]) #calculate histogram
        hist = cv2.normalize(hist).flatten()  #normalise histogram
        train_histograms[f] = hist  #store histogram 

    test_images = {}
    test_histograms = {}
    test_path = path+folder+"/test/"
    for f in os.listdir(test_path):
        im = cv2.imread(test_path+f)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        test_images[f] = im
        hist = cv2.calcHist([im], [0], None, [8], [0,255])
        hist = cv2.normalize(hist).flatten()
        test_histograms[f] = hist
    #OPENCV_METHODS = (("Correlation", cv2.cv.CV_COMP_CORREL),("Chi-Squared", cv2.cv.CV_COMP_CHISQR),("Intersection", cv2.cv.CV_COMP_INTERSECT)
    #(Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA))
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
    labels = {'s':'Severe', 'n':'Normal', 'm':'Morderate', 't':'Trace', 'i':'Mild'}
    print "=========   Results of " + folder + "experiment   ==============="
    for i, (k,v) in enumerate(img_results.items()): #for each test image and nearest neighbour
        label = ''
        if v[:1].lower() in labels:
            label = labels[v[:1].lower()]
        print "Patient "+str(i+1)+": " + k + " => " + v + " (" +label+")\n"   #print test image and nearest neighbour
    print "\n\n"


# In[ ]:




# 

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



