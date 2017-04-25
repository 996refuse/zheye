#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from PIL import Image, ImageFont, ImageDraw
import numpy as np

import os 
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import keras
model = keras.models.load_model(path +'/zheyeV3.keras')
#model = keras.models.load_model(path +'/zheye2.keras')
'''
************************************************************************
Recognizing...
'''
def Recognizing(fn):
    im = Image.open(fn)
    im = centerExtend(im, radius=20)

    vec = img2vec(im).copy()
    Y = []
    for i in range(vec.shape[0]):
        for j in range(vec.shape[1]):
            if vec[i][j] <= 200:
                Y.append([i, j])

    gmm = GaussianMixture(n_components=7, covariance_type='tied', reg_covar=1e2, tol=1e3, n_init=9)
    gmm.fit(Y)
    k_means_cluster_centers = np.sort(gmm.means_, axis=0)

    points = []

    for i in range(7):
        scoring = 0.0
        for w_i in range(3):
            for w_j in range(3):
                p_x = k_means_cluster_centers[i][0] -1 +w_i
                p_y = k_means_cluster_centers[i][1] -1 +w_j

                cr = crop(im, p_x, p_y, radius=20)
                cr = cr.resize((40, 40), Image.ANTIALIAS)

                X = np.asarray(cr.convert('L'), dtype='float')
                X = (X.astype("float") - 180) /200

                x0 = np.expand_dims(X, axis=0)
                x1 = np.expand_dims(x0, axis=3)

                global model
                if model.predict(x1)[0][0] < 0.5:
                    scoring += 1

        if scoring > 4:
            points.append((k_means_cluster_centers[i][0], k_means_cluster_centers[i][1]))
    return points

'''
************************************************************************
RandomGenerateOneFile()
'''
def crop(im, y, x, radius = 20):
    return im.crop((x-radius, y-radius, x+radius, y+radius))

def PaintPoint(image, points=[]):
    #print(points)
    im = image.copy()
    bgdr = ImageDraw.Draw(im)
    for y, x in points:
        bgdr.ellipse((x-6, y-6, x+6, y+6), fill ="red", outline ='red')
    return im

def Paint2File(contents, fn):
    '''
    contents = [(起始位置x,起始位置y,旋转角度,汉字), ]
    '''    
    background = Image.new("RGBA", (400, 88), (255,255,255,255))
    
    global path
    font = ImageFont.truetype(path + "/Kaiti-SC-Bold.ttf", 72)
    
    for c in contents:
        axis_x    = c[0]
        axis_y    = c[1]
        angle     = c[2]
        character = c[3]
        
        im = Image.new("RGBA", (72, 82), (0, 0, 0, 0))
        dr = ImageDraw.Draw(im)
        dr.text((0, 0), character, font=font, fill="#000000")
        
        
        fore = im.rotate(angle, expand=1);
        background.paste(fore, (axis_x, axis_y), fore)
        
    #uncomment to save to file
    #background.save(fn)
    return background

from random import randint, choice
from math import sin, cos, radians, fabs

def randomGB2312():
    '''
    来自 http://blog.3gcnbeta.com/2010/02/08/python-%E9%9A%8F%E6%9C%BA%E7%94%9F%E6%88%90%E4%B8%AD%E6%96%87%E7%9A%84%E4%BB%A3%E7%A0%81/
    有bug
    '''
    head = randint(0xB0, 0xDF)
    body = randint(0xA, 0xF)
    tail = randint(0, 0xF)
    val = ( head << 0x8 ) | (body << 0x4 ) | tail
    c = '%x' % val
    try:
        #return str.decode('hex').decode('gb2312')
        # python3 Compatible
        return bytes.fromhex(c).decode('gb2312')
    except:
        return randomGB2312()

def RandomGenerateOneFile(characters=[]):
    choices = [x for x in range(-20, 20)] + [x for x in range(-180, -160)] + [x for x in range(160, 180)]

    l = []
    
    if characters == []:
        length = 7
    else:
        length = len(characters)
    for i in range(length):
        
        angle = choice(choices)
        f = 0
        if angle <= 20 and angle >= -20:
            f = 1
        #汉字正 设置为1
        
        rad   = radians(angle)
        height = fabs( sin(rad) * 72 ) + fabs( cos(rad) * 82 )
        width  = fabs( sin(rad) * 82 ) + fabs( cos(rad) * 72 )
        
        x = i * 54 + randint(-3, 3)
        
        rg = int((88 - height)/2)
        y = randint(rg-3, rg+3)
        
        if characters == []:
            character = randomGB2312()
        else:
            character = characters[i]
        
        centerX = x + width/2
        centerY = y + height/2
        
        c = (x, y, angle, character, centerX, centerY, f)
        l.append(c)
    return Paint2File(l, '_'.join([  str(int(i[4]))  +'-'+  str(int(i[5]))  +'-'+  str(int(i[6]))  for i in l]) + '.png'), l

'''
************************************************************************
Training()
'''


def showAscii(x):
    import sys
    for i in x:
        for j in i:
            #if j > 0:
            if j > 200:
                sys.stdout.write('+')
            else:
                sys.stdout.write(' ')
        print

def vecCut(x1, center_x , center_y, radius = 20):
    return x1[center_y - radius:center_y + radius, center_x - radius:center_x + radius]

def centerExtend(im, width=400, height=88, radius=20):
    x1 = np.full((height+radius+radius, width+radius+radius), 255, dtype='uint8')
    x2 = np.asarray(im.convert('L'))
    x1[radius:radius+height,radius:radius+width] = x2
    return Image.fromarray(x1, 'L')
     
                 
def img2vec(im, width=400, height=88, radius=20):
    return np.asarray(im.convert('L'));

import scipy.ndimage

def Seven(ret):
    #迭代返回 r/2*r/2 点阵 正倒 
    for i in ret[1]:
        x = img2vec(ret[0], int(i[4]), int(i[5]), 400, 88)
        r = scipy.ndimage.zoom(x, 0.5, order=0)
        #showAscii(r)
        yield r, i[6]

if __name__ == '__main__':
    RandomGenerateOneFile()


