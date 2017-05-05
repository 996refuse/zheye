#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from PIL import Image, ImageFont, ImageDraw
import numpy as np

from random import randint, choice
from math import sin, cos, radians, fabs

import os 
dir_path = os.path.dirname(os.path.realpath(__file__)) 

def crop(im, y, x, radius = 20):
    return im.crop((x-radius, y-radius, x+radius, y+radius))

def PaintPoint(image, points=[]):
    im = image.copy()
    bgdr = ImageDraw.Draw(im)
    for y, x in points:
        bgdr.ellipse((x-3, y-3, x+3, y+3), fill ="red", outline ='red')
    return im

def RandomGenerateOneChar(y=None, character=None, radius=20):
    '''
    y == 1 汉字正
    y ==-1 汉字倒
    radius < 50
    '''
    choices = range(-30, 30) + range(-180, -150) + range(150, 180)
    
    angle = choice(choices)
    if y != None:
        while (angle <= 30 and angle >= -30) == (y == -1):
            angle = choice(choices)
    else:
        y = -1
        if angle <= 30 and angle >= -30:
            y = 1
    
    rad = radians(angle)
    if character == None:
        character = RandomGB2312()

    background = Image.new("RGBA", (160, 160), (255,255,255,255))
    
    im = Image.new("RGBA", (72, 82), (0, 0, 0, 0))
    global dir_path
    font = ImageFont.truetype(dir_path + "/Kaiti-SC-Bold.ttf", 72)
    
    dr = ImageDraw.Draw(im)
    dr.fontmode = "1"
    dr.text((0, 0), character, font=font, fill="#000000")
    
    fore = im.rotate(angle, expand=1)
    width, height = fore.size
    
    scale = np.random.uniform(0.8, 1.2)
    fore = fore.resize((int(width *scale), int(height*scale)), Image.ANTIALIAS)
    width, height = fore.size
    
    background.paste(fore, (80 - width/2 + randint(-10, 10), 80 -10*y - height/2 + randint(-10, 10)), fore)
    return background.crop((80-radius, 80-radius, 80+radius, 80+radius))

def RandomGB2312():
    '''
    来自
    http://blog.3gcnbeta.com/2010/02/08/
    python-%E9%9A%8F%E6%9C%BA%E7%94%9F%E6%88%90%E4%B8%AD%E6%96%87%E7%9A%84%E4%BB%A3%E7%A0%81/
    
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
        return RandomGB2312()

def Img2Vec(im):
    return np.asarray(im.convert('L'))

def Vec2Ascii(x):
    import sys
    for i in x:
        for j in i:
            #if j > 0:
            if j > 200:
                sys.stdout.write('+')
            else:
                sys.stdout.write(' ')
        print

def CenterExtend(im, width=400, height=88, radius=20):
    x1 = np.full((height+radius+radius, width+radius+radius), 255, dtype='uint8')
    x2 = np.asarray(im.convert('L'))
    x1[radius:radius+height,radius:radius+width] = x2
    return Image.fromarray(x1, 'L')

if __name__ == '__main__':
    pass
