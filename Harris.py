# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:17:03 2022

Harris corner detector:

@author: guido
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import signal
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pdb
from copy import copy, deepcopy

def Harris_pixel(x, y, gray, k = 0.04, Sobel =True):
    
    if(not Sobel):
        # this is as specified in the article (no Sobel)
        dx = (gray[y, x+1] - gray[y, x-1]) / 2
        dy = (gray[y+1, x] - gray[y-1, x]) / 2
    else:
        # Sobel
        dx = ((gray[y-1, x+1] - gray[y-1, x-1])+2*(gray[y, x+1] - gray[y, x-1])+(gray[y+1, x+1] - gray[y+1, x-1])) / 8
        dy = ((gray[y+1, x-1] - gray[y-1, x-1])+2*(gray[y+1, x] - gray[y-1, x])+(gray[y+1, x+1] - gray[y-1, x+1])) / 8
        
    
    dx2 = dx * dx
    dy2 = dy * dy
    dxdy = dx * dy
    
    tr = dx2 + dy2
    det = dx2 * dy2 - dxdy * dxdy
    
    M = np.asarray([[dx2, dxdy],[dxdy, dy2]])
    ev = np.linalg.eigvals(M)
    alpha = ev[0]
    beta = ev[1]
    
    H = abs(det - k * tr*tr)
    
    return [H, alpha, beta, tr, det, dx, dy]
    
def non_maximum_suppression(Harris, dist = 1, threshold_factor = 0.05):
    
    max_H = np.max(Harris[:])
    threshold = threshold_factor * max_H
    
    Corners = np.zeros([Harris.shape[0], Harris.shape[1]])
    for x in range(1, Harris.shape[1]-1):
        for y in range(1, Harris.shape[0]-1):
            if Harris[y,x] >= threshold:
                Harris[y-dist:y+dist+1, x-dist:x+dist+1] = 0 # suppress neighbors
                Corners[y,x] = 1 # indicate corner
    return Corners

# Vectorized version from: # From: https://muthu.co/harris-corner-detector-implementation-in-python/

def gradient_x(imggray):
    ##Sobel operator kernels.
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    return signal.convolve2d(imggray, kernel_x, mode='same')

def gradient_y(imggray):
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return signal.convolve2d(imggray, kernel_y, mode='same')
    
def get_Harris_vectorized(filename, IMAGE_PATCHES = False):
    # load the BGR color image:
    BGR = cv2.imread(filename)
    gray = cv2.cvtColor(BGR, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    width = gray.shape[1]
    if(height * width >= 1E5):
        gray = cv2.resize(gray, (int(width/4), int(height/4)))
        BGR = cv2.resize(BGR, (int(width/4), int(height/4)))
        height = gray.shape[0]
        width = gray.shape[1]
    
    gray_float = gray.astype(float) / 255.0
    
    I_x = gradient_x(gray_float)
    I_y = gradient_y(gray_float)
    
    Ixx = scipy.ndimage.gaussian_filter(I_x**2, sigma=1)
    Ixy = scipy.ndimage.gaussian_filter(I_y*I_x, sigma=1)
    Iyy = scipy.ndimage.gaussian_filter(I_y**2, sigma=1)
    
    k = 0.04
    
    # determinant
    detA = Ixx * Iyy - Ixy ** 2
    # trace
    traceA = Ixx + Iyy
        
    Harris = detA - k * traceA ** 2
    
    #    plt.figure()
    #    plt.imshow(detA)
    #    plt.colorbar()
    #    plt.title('Determinant')

    plt.figure()
    plt.imshow(Harris)
    plt.colorbar()
    plt.title('Harris response')

    Corners = non_maximum_suppression(Harris)
    Corners = scipy.ndimage.binary_dilation(Corners)
    BGR[Corners > 0] = [0,0,255]
    BGR[Harris < 0] = [255,0,0]
    cv2.imshow('Corners', BGR)
    
    return [Harris, Ixx, Iyy, Ixy, BGR, detA, traceA, I_x, I_y]

def get_Harris_response(filename):
    # load the BGR color image:
    BGR = cv2.imread(filename)
    gray = cv2.cvtColor(BGR, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    width = gray.shape[1]
    gray = cv2.resize(gray, (int(width/4), int(height/4)))
    BGR = cv2.resize(BGR, (int(width/4), int(height/4)))
    height = gray.shape[0]
    width = gray.shape[1]
    
    # floating point values in here:
    Harris = np.zeros([height, width])
    gray_float = gray.astype(float)
    
    #fig, ax = plt.subplots()
    
    # loop over the image, ignoring a 1-pixel border in which the gradients are not defined
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            [H, alpha, beta, tr, det, dx, dy] = Harris_pixel(x, y, gray_float)
            Harris[y,x] = H
            
#            if(det != 0):
#                patch = OffsetImage(gray_float[y-1:y+2, x-1:x+2], zoom=3)  
#                ab = AnnotationBbox(patch, (alpha, beta), frameon=False)
#                ax.add_artist(ab)
    
    Corners = non_maximum_suppression(Harris, threshold_factor = 0.20)
    Corners = scipy.ndimage.binary_dilation(Corners)
    
    plt.figure()
    plt.imshow(Harris)
    plt.colorbar()
    
    Harris = (Harris / np.max(Harris[:])) * 255
    
    Harris_image = Harris.astype(np.uint8)
    cv2.imshow('Harris', Harris_image);
    
    
    BGR[Corners > 0] = [0,0,255]
    cv2.imshow('Corners', BGR)


def plot_patches(n_patches = 1000):
    ''' Generate 3x3 patches and plot them in the Harris eigenvalue space.
    '''
    # https://stackoverflow.com/questions/15075239/the-harris-stephens-corner-detection-algorithm-determinant-always-0-zero
    
    fig, ax = plt.subplots()
    
    for i in range(n_patches):
        
        r = np.random.rand(1)
        r = r[0]
        if(r < 0.1):
            gray = np.random.rand(1) * np.ones([3,3])
        elif(r < 0.2):
            gray = np.matmul(np.random.rand(3,1),  np.random.rand(1,3))
        elif(r < 0.3):
            gray = np.matmul(np.random.rand(3,1),  np.random.rand(1,3))
            gray = np.transpose(gray)
        elif(r < 0.4):
            gray = np.random.rand(3,3)
            for i in range(3):
                for j in range(i, 3):
                    gray[j,i] = gray[i,j]
        elif(r < 0.5):
            gray = np.random.rand(3,3)
            for i in range(3):
                for j in range(i, 3):
                    gray[i,j] = gray[j,i]
        elif(r < 0.6):
            gray = np.repeat(np.random.rand(1,3), 3, 0)
        elif(r < 0.7):
            gray = np.repeat(np.random.rand(1,3), 3, 0)
            gray = np.transpose(gray)
        else:
            gray = np.random.rand(3,3)

        dx = (gray[1, 2] - gray[1, 0]) / 2
        dy = (gray[2, 1] - gray[0, 1]) / 2
        
        dx2 = dx * dx
        dy2 = dy * dy
        dxdy = dx * dy
        M = np.asarray([[dx2, dxdy],[dxdy, dy2]])
        
        ev = np.linalg.eigvals(M)
        alpha = ev[0]
        beta = ev[1]
        #plt.plot(alpha, beta, 'kx')
        patch = OffsetImage(gray, zoom=10)  
        ab = AnnotationBbox(patch, (alpha, beta), frameon=False)
        ax.add_artist(ab)

def plot_eigenvalues(Harris, Ixx, Iyy, Ixy, BGR, k=0.04, IMAGE_PATCHES=False):

    height = BGR.shape[0]
    width = BGR.shape[1]
    
    threshold = 5E3 / (height* width)
    print(f'Threshold = {threshold}')
    
    fig, ax = plt.subplots()
    alpha = np.arange(0, 7, 0.01)
    beta = np.arange(0, 7, 0.01)
    [X,Y] = np.meshgrid(alpha, beta)
    R = X * Y - k * (X+Y)**2
    plt.figure()
    levels = np.arange(-3, 6, 1)
    cs = plt.contour(X, Y, R, levels)
    plt.clabel(cs, levels)
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
        
            dx2 = Ixx[y,x]
            dy2 = Iyy[y,x]
            dxdy = Ixy[y,x]
            
            M = np.asarray([[dx2, dxdy],[dxdy, dy2]])
            ev = np.linalg.eigvals(M)
            alpha = ev[0]
            beta = ev[1]
            
            det = dx2 * dy2 - dxdy * dxdy
#            if(detA[y,x] != 0):
#                print(f'detA[{y},{x}] = {detA[y,x]}, det = {det}')
            
            r = np.random.rand(1)
            if(det != 0 and r < threshold):
                plt.plot(alpha, beta, 'kx')
                
                if(IMAGE_PATCHES):
                    Patch = BGR[y-1:y+2, x-1:x+2]
                    RGB = deepcopy(Patch)
                    RGB[:,:,0] = Patch[:,:,2]
                    RGB[:,:,2] = Patch[:,:,0]
                    patch = OffsetImage(RGB, zoom=3)  
                    ab = AnnotationBbox(patch, (alpha, beta), frameon=False)
                    ax.add_artist(ab)
    plt.title('Eigenvalues')
    
# plot_patches()

# get_Harris_response('bebop_flowers_1.jpg')
# get_Harris_response('chess_board.jpg')
# get_Harris_vectorized('chess_board.jpg')
# get_Harris_vectorized('bebop_flowers_1.jpg')
#get_Harris_vectorized('flower.png')
#get_Harris_vectorized('bebop.png')

[Harris, Ixx, Iyy, Ixy, BGR, detA, traceA, I_x, I_y] = get_Harris_vectorized('bebop.png')
#plot_eigenvalues(Harris, Ixx, Iyy, Ixy, BGR, k=0.04)
