# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 12:40:56 2019

@author: essie-adm-daniele.p
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import cv2

imgIR1 = cv2.imread('ThermalInfrared1.jpg',0)
imgIR2 = cv2.imread('ThermalInfrared2.jpg',0)

#tazza
edgesIR1Cup = cv2.Canny(imgIR1,130,500)
edgesIR1Suit = cv2.Canny(imgIR1,40,100)
edgesIR1Ref = cv2.Canny(imgIR1,40,100)
edgesIR2 = cv2.Canny(imgIR2,80,120)

sIR1Cup = np.linspace(0, 2*np.pi, 400)
x1 = 50 + 50*np.cos(sIR1Cup)
y1 = 120 + 50*np.sin(sIR1Cup)
init1Cup = np.array([x1, y1]).T

sIR1Suit = np.linspace(0, 2*np.pi, 400)
x2 = 255 + 80*np.cos(sIR1Suit)
y2 = 180 + 90*np.sin(sIR1Suit)
init1Suit = np.array([x2, y2]).T

sIR1Ref = np.linspace(0, 2*np.pi, 400)
x3 = 380 + 50*np.cos(sIR1Ref)
y3 = 140 + 100*np.sin(sIR1Ref)
init1Ref = np.array([x3, y3]).T

snake1Cup = active_contour(gaussian(imgIR1, 3),init1Cup, alpha=0.01, beta=25, gamma=0.001)
snake1CanCup = active_contour(gaussian(edgesIR1Cup, 2),init1Cup, alpha=0.1, beta=40, gamma=0.02)

snake1Suit = active_contour(gaussian(imgIR1, 3),init1Suit, alpha=0.02, beta=25, gamma=0.001)
snake1CanSuit = active_contour(gaussian(edgesIR1Suit, 2),init1Suit, alpha=0.1 , beta=40, gamma=0.02)

snake1Ref = active_contour(gaussian(imgIR1, 3),init1Ref, alpha=0.02, beta=25, gamma=0.001)
snake1CanRef = active_contour(gaussian(edgesIR1Ref, 2),init1Ref, alpha=0.1 , beta=40, gamma=0.02)

plt.subplots(figsize=(10,7))
plt.subplot(2,2,2)
plt.imshow(edgesIR1Cup, cmap=plt.cm.gray)
plt.plot(init1Cup[:, 0], init1Cup[:, 1], '--r', lw=3)
plt.plot(snake1CanCup[:, 0], snake1CanCup[:, 1], '-b', lw=3)
plt.subplot(2,2,4)
plt.imshow(edgesIR1Suit, cmap=plt.cm.gray)
plt.plot(init1Suit[:, 0], init1Suit[:, 1], '--r', lw=3)
plt.plot(snake1CanSuit[:, 0], snake1CanSuit[:, 1], '-b', lw=3)
plt.subplot(2,2,3)
plt.imshow(edgesIR1Ref, cmap=plt.cm.gray)
plt.plot(init1Ref[:, 0], init1Ref[:, 1], '--r', lw=3)
plt.plot(snake1CanRef[:, 0], snake1CanRef[:, 1], '-b', lw=3)
plt.subplot(2,2,1)
plt.imshow(imgIR1, cmap=plt.cm.gray)
plt.plot(init1Cup[:, 0], init1Cup[:, 1], '--r', lw=3)
plt.plot(snake1Cup[:, 0], snake1Cup[:, 1], '-b', lw=3)
plt.plot(init1Suit[:, 0], init1Suit[:, 1], '--r', lw=3)
plt.plot(snake1Suit[:, 0], snake1Suit[:, 1], '-b', lw=3)
plt.plot(init1Ref[:, 0], init1Ref[:, 1], '--r', lw=3)
plt.plot(snake1Ref[:, 0], snake1Ref[:, 1], '-b', lw=3)

imgB1 = cv2.imread('Bird1.jpg',0)
imgB2 = cv2.imread('Bird2.jpg',0)
imgB3 = cv2.imread('Bird3.jpg',0)

#tazza
edgesIB1 = cv2.Canny(imgB1,130,500)
edgesIB2 = cv2.Canny(imgB2,40,100)
edgesIB3 = cv2.Canny(imgB3,40,100)

sIB1 = np.linspace(0, 2*np.pi, 200)
xb1 = 30 + 20*np.cos(sIB1)
yb1 = 45 + 35*np.sin(sIB1)
initIB1 = np.array([xb1, yb1]).T

sIB2 = np.linspace(0, 2*np.pi, 400)
xb2 = 255 + 80*np.cos(sIB2)
yb2 = 180 + 90*np.sin(sIB2)
initIB2 = np.array([xb2, yb2]).T

sIB3 = np.linspace(0, 2*np.pi, 400)
xb3 = 380 + 50*np.cos(sIB3)
yb3 = 140 + 100*np.sin(sIB3)
initIB3 = np.array([xb3, yb3]).T

snakeIB1 = active_contour(gaussian(imgB1, 3),initIB1, alpha=0.01, beta=25, gamma=0.001)
snakeCanIB1 = active_contour(gaussian(edgesIB1, 2),initIB1, alpha=0.01, beta=40, gamma=0.02)

snakeIB2 = active_contour(gaussian(imgB1, 3),initIB2, alpha=0.01, beta=25, gamma=0.001)
snakeCanIB2 = active_contour(gaussian(edgesIB2, 2),initIB2, alpha=0.1, beta=40, gamma=0.02)

snakeIB3 = active_contour(gaussian(imgB3, 3),initIB3, alpha=0.01, beta=25, gamma=0.001)
snakeCanIB3 = active_contour(gaussian(edgesIB3, 2),initIB3, alpha=0.1, beta=40, gamma=0.02)

plt.subplots(figsize=(10,7))
plt.subplot(1,2,2)
plt.imshow(edgesIB1, cmap=plt.cm.gray)
plt.plot(initIB1[:, 0], initIB1[:, 1], '--r', lw=3)
plt.plot(snakeIB1[:, 0], snakeIB1[:, 1], '-b', lw=3)
plt.subplot(1,2,1)
plt.imshow(imgB1, cmap=plt.cm.gray)
plt.plot(initIB1[:, 0], initIB1[:, 1], '--r', lw=3)
plt.plot(snakeIB1[:, 0], snakeIB1[:, 1], '-b', lw=3)

def EqHistogram(img):
    equ = cv2.equalizeHist(img)
    return equ

