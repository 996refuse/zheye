#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
************************************************************************
RandomGenerateOneFile()
'''

from PIL import Image, ImageFont, ImageDraw
import numpy as np

def PaintPoint(image, points=[]):
    im = image.copy()
    bgdr = ImageDraw.Draw(im)
    for y, x in points:
        bgdr.ellipse((x-3, y-3, x+3, y+3), fill ="blue", outline ='blue')
    return im

def Paint2File(contents, fn):
    '''
    contents = [(起始位置x,起始位置y,旋转角度,汉字), ]
    '''    
    background = Image.new("RGBA", (400, 88), (255,255,255,255))
    
    font = ImageFont.truetype("./Kaiti-SC-Bold.ttf", 72)
    
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
    str = '%x' % val
    try:
        return str.decode('hex').decode('gb2312')
    except:
        return randomGB2312()

def RandomGenerateOneFile():
    choices = range(-20, 20) + range(-180, -160) + range(160, 180)

    l = []
    for i in range(7):
        angle = choice(choices)
        f = 0
        if angle <= 20 and angle >= -20:
            f = 1
        #汉字正 设置为1
        
        rad   = radians(angle)
        height = fabs( sin(rad) * 72 ) + fabs( cos(rad) * 82 )
        width  = fabs( sin(rad) * 82 ) + fabs( cos(rad) * 72 )
        
        x = i * 54 + randint(-9, 9)
        
        rg = int((88 - height)/2)
        y = randint(rg-3, rg+3)
        character = randomGB2312()
        
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
    return x1;
                 
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

'''
def Training():
    from pybrain.supervised.trainers import BackpropTrainer
    from pybrain.tools.shortcuts import buildNetwork
    net = buildNetwork(400, 600, 30, 1, bias=True)

    from pybrain.datasets import SupervisedDataSet
    DS = SupervisedDataSet(400, 1)

    for uuid in range(1000):
        ret = RandomGenerateOneFile()
        for i in Seven(ret):
            #showAscii(i[0]),
            #print i[1]
            DS.appendLinked( [0 if x < 200 else 1 for x in i[0].ravel().tolist()], [i[1]] )

    trainer = BackpropTrainer(net, DS)

    trainer.trainUntilConvergence(maxEpochs=30)

    import pickle
    pickle.dump( net, open( "net.p", "wb" ) )
'''

if __name__ == '__main__':
    RandomGenerateOneFile()
