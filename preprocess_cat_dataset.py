# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 14:02:41 2017

@author: SalaFeng-
"""
import cv2
import glob
import math
import sys

def rotateCoords(coords, center, angleRadians):
    angleRadians = -angleRadians
    xs, ys = coords[::2], coords[1::2]
    newCoords = []
    n = min(len(xs), len(ys))
    i = 0
    centerX = center[0]
    centerY = center[1]
    cosAngle = math.cos(angleRadians)
    sinAngle = math.sin(angleRadians)
    while i < n:
        xOffset = xs[i] - centerX
        yOffset = ys[i] - centerY
        newX = xOffset * cosAngle - yOffset * sinAngle + centerX
        newY = xOffset * sinAngle + yOffset * cosAngle + centerY
        newCoords += [newX, newY]
        i += 1
    return newCoords

def getcatface(image,coords):
    x1, y1 = coords[0], coords[1]
    x2, y2 = coords[2], coords[3]
    mouthX = coords[4]
    if x1 > x2 and y1 < y2 and mouthX > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    eyesCenter = (0.5 * (x1 + x2),0.5 * (y1 + y2))
    eyesDeltaX = x2 - x1
    eyesDeltaY = y2 - y1
    eyesAngleRadians = math.atan2(eyesDeltaY, eyesDeltaX)
    eyesAngleDegrees = eyesAngleRadians * 180.0 / math.pi
    rotation = cv2.getRotationMatrix2D(eyesCenter, eyesAngleDegrees, 1.0)
    imageSize = image.shape[1::-1]
    straight = cv2.warpAffine(image, rotation, imageSize,borderValue=(128, 128, 128))
    newCoords = rotateCoords(coords, eyesCenter, eyesAngleRadians)
    w = abs(newCoords[16] - newCoords[6])
    
    # Put the center point between the eyes at (0.5, 0.4) in
	# proportion to the entire face.    
    h = w
    minX = eyesCenter[0] - w/2
    if minX < 0:
        w += minX
        minX = 0
    minY = eyesCenter[1] - h*2/5
    if minY < 0:
        h += minY
        minY = 0
    crop = straight[int(minY):int(minY+h), int(minX):int(minX+w)]
    return crop
 
    

def main():
    for imagepath in glob.glob('cat_dataset/*.jpg'):
        image = cv2.imread(imagepath)
        input =open('%s.cat'%imagepath,'r')
        coords  =[int(i) for i in input.readline().split()[1:]]
        crop = getcatface(image,coords)
        if crop is None:
            print('Failed to preprocess image at %s.'%imagepath)
            continue
        h, w, colors = crop.shape
        if min(h,w) >= 64:
            path = imagepath.replace("cat_dataset","cats_64x64")
            out =cv2.resize(crop,(64,64),interpolation=cv2.INTER_AREA)
            cv2.imwrite(path, out)
        if min(h,w) >= 128:
            path = imagepath.replace("cat_dataset","cats_128x128")
            out =cv2.resize(crop,(128,128))
            cv2.imwrite(path, out)
        print (imagepath,coords)
        
def calculate():
    img_all=0   #9858
    img_64=0    #9302
    img_128=0   #6443
    for imagepath in glob.glob('cat_dataset/*.jpg'):
        img_all +=1
    for imagepath in glob.glob('cats_64x64/*.jpg'):
        image =cv2.imread(imagepath)
        print(image.shape)
        
        img_64 +=1
    for imagepath in glob.glob('cats_128x128/*.jpg'):
        img_128 +=1
    print(img_all,img_64,img_128)

main()