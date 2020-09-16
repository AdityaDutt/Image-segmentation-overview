import cv2, os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.color import rgb2gray

NumClusters = 10
Sum_of_squared_distances = []


# names = ['Bird1.jpg', 'Bird2.jpg', 'Bird3.jpg', 'ThermalInfrared1.jpg', 'ThermalInfrared2.jpg', 'Squall1.jpeg','Squall2.jpeg','Squall3.jpeg','Squall4.jpeg','Squall5.jpeg']
currDir = os.getcwd() + "/CCL/"
names = []
import os
for root, dirs, files in os.walk(currDir, topdown=False):
   for name in files:
       if name.find(".png") != - 1:
           names.append(name)
print(names)

for ind in range(len(names)) :
    filename = names[ind] # Set filename

    # Create a directory to save clusters of input image
    dir = os.getcwd()
    # Comment this part if you don't want to save image
    dirName = filename
    dirName = (dirName.split("."))[0]
    dir = dir + "/CCLKNN/" + dirName
    os.mkdir(dir)
    ####
    filename = "/Users/adityadutt/Documents/ENV/Project2/CCL/" + filename
    pic = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    NumChannel = len(pic.shape)
    Sum_of_squared_distances = []

    if NumChannel == 2 : # 2 channel image (Gray scale image)

        # Equalize image
        pic = cv2.equalizeHist(pic)
        pic = pic/255  # Scale the pixel values between 0 and 1    
        
        # Reshape (row, col) into (row*col, 1)
        pic_n = pic.reshape(pic.shape[0]*pic.shape[1], 1)

        # Try k means clustering for different numbers of clusters
        for i in range(NumClusters) :

            kmeans = KMeans(n_clusters=i+1, random_state=0).fit(pic_n) # Run k means clustering
            Sum_of_squared_distances.append(kmeans.inertia_) # Find WCSS and store it
            pic2show = kmeans.cluster_centers_[kmeans.labels_] # Find cluster centroids
            cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1]) # Reshape image again in 3 channel image
            cv2.imwrite(dir+"/cluster"+str(i+1)+".png", cluster_pic*255)
            # plt.imshow(cluster_pic, cmap='gray')
            # plt.savefig(dir+"/cluster"+str(i+1)+".png") # Save image
            # plt.show()

        # Plot WCSS graph
        xaxis = np.arange(1,11,1) 
        yaxis = np.array(Sum_of_squared_distances)
        # print(xaxis, yaxis)
        plt.cla()
        plt.clf()
        plt.close()

        plt.plot(xaxis,yaxis)
        plt.title("Finding optimal number of clusters using Elbow method")
        plt.xlabel("Number of clusters")
        plt.ylabel("WCSS")
        # Save the WCSS plot
        plt.savefig(dir+"/WCSS.png")
        plt.cla()
        plt.clf()
        plt.close()
        # plt.show()


    elif NumChannel == 3 :  # Color image (No nned of histogram equalization)

        pic = pic/255  # Scale the pixel values between 0 and 1    
        
        # Reshape (row, col, height) into (row*col, height)
        pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])

        # Try k means clustering for different numbers of clusters
        for i in range(NumClusters) :

            kmeans = KMeans(n_clusters=i+1, random_state=0).fit(pic_n) # Run k means clustering
            Sum_of_squared_distances.append(kmeans.inertia_) # Find WCSS and store it
            pic2show = kmeans.cluster_centers_[kmeans.labels_] # Find cluster centroids
            cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2]) # Reshape image again in 3 channel image
            cv2.imwrite(dir+"/cluster"+str(i+1)+".png", cluster_pic*255)
            # plt.imshow(cluster_pic)
            # plt.show()

            # plt.imshow(cluster_pic, cmap='gray')
            # plt.savefig(dir+"/cluster"+str(i+1)+".png") # Save image
            # plt.show()

        # Plot WCSS graph
        xaxis = np.arange(1,11,1) 
        yaxis = np.array(Sum_of_squared_distances)
        plt.cla()
        plt.clf()
        plt.close()

        plt.plot(xaxis,yaxis)
        plt.title("Finding optimal number of clusters using Elbow method")
        plt.xlabel("Number of clusters")
        plt.ylabel("WCSS")
        # Save the WCSS plot
        plt.savefig(dir+"/WCSS.png")
        plt.cla()
        plt.clf()
        plt.close()
        # plt.show()