import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import os, cv2
from time import time


def change_color_fuzzycmeans(cluster_membership, clusters):
    img = []
    for pix in cluster_membership.T:
        img.append(clusters[np.argmax(pix)])
    return img

def readimage(filepath):

    list_img = []
    img = cv2.imread(filepath)
    shape = img.shape
    rgb_img = img.reshape((img.shape[0] * img.shape[1], 3))
    list_img.append(rgb_img)
        
    return [list_img, shape]



################################################################## 
######################## Code starts here ######################## 
################################################################## 

def RunFuzzyKnn(filepath) :

    list_img, org_shape = readimage(filepath)
    n_data = len(list_img)
    clusters = np.arange(2,15,1)

    currDir = os.getcwd()
    dir = currDir + "/" + "FuzzyClusters/"
    FPC = []

    filename = (filepath.split("/") )[-1]
    filename = (filename.split("."))[0]

    os.mkdir(dir+filename) # create directroy for image clusters
    for index,rgb_img in enumerate(list_img):

        img = np.reshape(rgb_img, org_shape).astype(np.uint8)
        shape = np.shape(img)
        
        # looping every cluster     
        print('Image '+str(index+1))
        for i,cluster in enumerate(clusters):
                
            # Fuzzy C Means
            new_time = time()
            
            print("i: ",i," cluster: ",cluster)
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            rgb_img.T, cluster, 2, error=0.005, maxiter=1000, init=None,seed=42)

            FPC.append(fpc)
            print("FPC",fpc)

            new_img = change_color_fuzzycmeans(u,cntr)
            print("Length of clusters: ", len(cntr))
            fuzzy_img = np.reshape(new_img,shape).astype(np.uint8)
            
            thresh = np.max(fuzzy_img) - 1
            # ret, seg_img = cv2.threshold(fuzzy_img, thresh, 255, cv2.THRESH_BINARY)
            ret, seg_img = cv2.threshold(fuzzy_img,128,255,cv2.THRESH_BINARY)

            print('Fuzzy time for cluster',cluster)
            print(time() - new_time,'seconds')


            cv2.imwrite(dir+filename+"/Cluster"+str(cluster)+".png",fuzzy_img )


    x_axis = np.arange(2,15,1)
    y_axis = np.array(FPC)

    fig = plt.gcf()
    plt.plot(x_axis,y_axis)
    plt.title("FPC Plot")
    plt.xlabel("Number of clusters")
    plt.ylabel("FPC")

    plt.savefig(dir+filename+"/FPC.png") 
    plt.cla()
    plt.clf()
    plt.close


################################################
############## Run Fuzzy KNN Function ##########
################################################

filenames = ["Squall1.jpeg","Squall2.jpeg","Squall3.jpeg","Squall4.jpeg","Squall5.jpeg","Bird1.jpg","Bird2.jpg","Bird3.jpg","ThermalInfrared1.jpg","ThermalInfrared2.jpg"]
currDir = os.getcwd()

for i in range(len(filenames)) :
    RunFuzzyKnn(currDir +"/" + filenames[i])