
import cv2, os, sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pylab as plt

warnings.filterwarnings("ignore")


def hex_to_rgb(hex):
     hex = hex.lstrip('#')
     hlen = len(hex)
     return tuple(int(hex[i:i+int(hlen/3)], 16) for i in range(0, hlen, int(hlen/3) ))


### morphology

def image2binaryIm(Image):

    input_image = cv2.imread(Image,0)
    kernel = np.ones((5,5), np.uint8)       # set kernel as 5x5 matrix from numpy

    # Create erosion and dilation image from the original image

    input_image = 255 - input_image
    dilation_image = cv2.dilate(input_image, kernel, iterations=1)
    dilation_image= dilation_image.astype(np.uint8)

    # Convert to binary

    (thresh, im_bw) = cv2.threshold(dilation_image, 128, 255, cv2.THRESH_BINARY)
    
    dilation_image = cv2.erode(im_bw, kernel, iterations=1)

    return dilation_image




# This function returns 8 connected nieghbours of an element in the form of a list.

def Get_Neighbor_8_Connected(x_coordinate, y_coordinate, row, col, image):
    Neighbor = []
    FinalNeighbors = []
    Neighbor.append([x_coordinate, y_coordinate+1])
    Neighbor.append([x_coordinate, y_coordinate-1])
    Neighbor.append([x_coordinate+1, y_coordinate])
    Neighbor.append([x_coordinate-1, y_coordinate])
    Neighbor.append([x_coordinate+1, y_coordinate+1])
    Neighbor.append([x_coordinate-1, y_coordinate-1])
    Neighbor.append([x_coordinate-1, y_coordinate+1])
    Neighbor.append([x_coordinate+1, y_coordinate-1])

    for i in range(len(Neighbor)):
        x, y = Neighbor[i]
        if (x >= 0 and x < row and y >= 0 and y < col and image[x][y] > 0):
            FinalNeighbors.append([x, y])
    Neighbor = []

    return FinalNeighbors


# This function returns 4 connected nieghbours of an element in the form of a list.
def Get_Neighbor_4_Connected(x_coordinate, y_coordinate, row, col, image):
    Neighbor = []
    FinalNeighbors = []
    Neighbor.append([x_coordinate, y_coordinate+1])
    Neighbor.append([x_coordinate, y_coordinate-1])
    Neighbor.append([x_coordinate+1, y_coordinate])
    Neighbor.append([x_coordinate-1, y_coordinate])
    for i in range(len(Neighbor)):
        x, y = Neighbor[i]
        if (x >= 0 and x < row and y >= 0 and y < col and image[x][y] > 0):
            FinalNeighbors.append([x, y])
    Neighbor = []

    return FinalNeighbors


# This function returns 8 connected nieghbours of an element in matrix. It returns [-1,array of labels of neighbours] if all
# neighbours have no label till now and [0,max(labels)] if neighbours have labels.
def Get_Neighbor_Label(array, labels):
    CheckNeigbor = []
    for i in range(len(array)):
        x, y = array[i]
        CheckNeigbor.append(labels[x][y])
    if max(CheckNeigbor) == -1:
        return [1, 0]
    else:
        LabelledNeighbors = [x for x in CheckNeigbor if x != -1]
        LabelledNeighbors = list(set(LabelledNeighbors))
        if len(LabelledNeighbors) == 1:
            return [0, max(CheckNeigbor)]
        else:
            return [-1, LabelledNeighbors]


#########################################################################################################################
################################################    CODE STARTS HERE    #################################################
#########################################################################################################################
##                                                                                                                      #
## - This function takes as input a binary image, number of rows, number of columns and a parameter var.                #
## - If var is 4 then 4 connected component analysis is done. If var is 8 then 8 connected component analysis is done.  #
## - This function returns unique labels in image and the 2D image array with labels as it's elements.                  #
##                                                                                                                      #
#########################################################################################################################

'''
Example - A = [[1,0,1,0],
               [0,0,0,1],
               [1,1,1,1]]

          Image should be binary (0 and 1)
          1 is foreground
          0 is background

          row, col = A.shape

          To find 4 connected components  -  ccl(A, row, col, 4)
                                   Output - [ [0,1,2,3], [[1,0,2,0],
                                                          [0,0,0,3],
                                                          [3,3,3,3]]  ]

          To find 8 connected components  -  ccl(A, row, col, 8)
                                   Output - [ [0,1,2], [[1,0,2,0],
                                                        [0,0,0,2],
                                                        [2,2,2,2]]  ]
          
          Returns [list of labels, Image with labels as elements ]
'''

def ccl(ImageBinary, row, col, var, flag):

    SameLabels = []
    LabelCount = 50
    Labels = np.zeros((row, col)) - 1

    for i in range(row):
        for j in range(col):
            if ImageBinary[i][j] > 0:
                if var == 8:
                    n = Get_Neighbor_8_Connected(i, j, row, col, ImageBinary)
                else:
                    n = Get_Neighbor_4_Connected(i, j, row, col, ImageBinary)
                if len(n) > 0:
                    l = Get_Neighbor_Label(n, Labels)

                    if l[0] == 1:
                        LabelCount = LabelCount + 1
                        Labels[i][j] = LabelCount
                        for k in range(len(n)):
                            [c1, c2] = n[k]
                            Labels[c1, c2] = LabelCount

                    elif l[0] == 0:
                        Labels[i][j] = l[1]
                        for k in range(len(n)):
                            [c1, c2] = n[k]
                            Labels[c1, c2] = l[1]
                    elif l[0] == -1:
                        same = l[1]
                        SameLabels.append(same)
                        for m in range(len(same)):
                            Labels[Labels == same[m]] = min(same)
                        Labels[c1, c2] = min(same)
                elif len(n) == 0:
                    LabelCount = LabelCount + 1
                    Labels[i][j] = LabelCount

    Labels[Labels < 0] = 0
    LabelList = np.unique(Labels)
    LabelList = np.delete(LabelList, 0)
    LabelList = sorted(LabelList)
    RemoveIndex = []
    
    # Remove labels that are not present in image
    for i in range(len(LabelList)):
        num = (Labels == LabelList[i]).sum()
        if num == 0:
            RemoveIndex.append(LabelList[i])

    for i in range(len(RemoveIndex)):
        LabelList.remove(RemoveIndex[i])

    if flag == 1 :

        indices = []
        threshold = 100    
        RemoveIndex = []
        for i in range(len(LabelList)):
            lab = LabelList[i]
            indx, indy = np.where(Labels==lab)
            if len(indx) <= threshold :
                for j in range(len(indx)) :
                    x = indx[j]
                    y = indy[j]
                    indices.append([x,y])
        
        for j in range(len(indices)) :
            x, y = indices[j]
            ImageBinary[x][y] = 255

        ImageBinary[ImageBinary>0] = 255
        return ImageBinary

    Labels = Labels.astype(np.uint8)
    div = len(LabelList)
    col = 0
    
    from random import randint
    colors = []
    rgbCol = []
    for i in range(div):
        # while True :
        colors='#%06X' % randint(0, 0xFFFFFF)
        rgb = hex_to_rgb(colors)
        r, g, b = rgb
        # if r >= 200 and g >= 200 and b >= 200 :
        #     break
        rgbCol.append(rgb)

    # Convert to color image
    colorIm = cv2.cvtColor(Labels, cv2.COLOR_GRAY2BGR)

    for i in range(len(LabelList)) :
        color1, color2, color3 = rgbCol[i]
        lab = int(LabelList[i])
        b = colorIm[:,:,0]
        g = colorIm[:,:,1]
        r = colorIm[:,:,2]
        r[r==lab] = color1
        g[g==lab] = color2
        b[b==lab] = color3
        colorIm[:,:,0] = b
        colorIm[:,:,1] = g
        colorIm[:,:,2] = r

    # Return list of labels and image matrix where each element is label of each component

    return colorIm

def main(Im) :

    ImageBinary=image2binaryIm(Im)
    row=ImageBinary.shape[0]
    col=ImageBinary.shape[1]
    var=4
    OutputIm = ccl(ImageBinary, row, col, var, 1)
    OutputIm = ccl(OutputIm, row, col, var, 0)

    return OutputIm


# this is the part for running the code

currDir = os.getcwd()

filenames = ["Squall1.jpeg","Squall2.jpeg","Squall3.jpeg","Squall4.jpeg","Squall5.jpeg","Bird1.jpg","Bird2.jpg","Bird3.jpg","ThermalInfrared1.jpg","ThermalInfrared2.jpg"]
# filenames = ["ThermalInfrared2.jpg"]
currDir = os.getcwd()

for i in range(len(filenames)) :
    Im = currDir +"/"+ filenames[i]
    OutputIm = main(Im)
    name = filenames[i]
    name = name.split(".")
    name = name[0]
    outputDir = currDir+"/CCL/"+name+"_CCL_Morph.png"
    print(outputDir)
    cv2.imwrite(outputDir, OutputIm)