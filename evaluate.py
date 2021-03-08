import sklearn.mixture
from PIL import Image, ImageFont, ImageDraw
import numpy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def CAPTCHA_to_data(filename):
    '''
    convert CAPTCHA image to 7 chinese character image data.
    kind of slow because of GMM iteration.
    return a 7 * 40 * 40 array
    '''
    
    width=400
    height=88
    padding=20
    padding_color = 249
    
    captcha = Image.open(filename)

    bg = numpy.full((height+padding*2, width+padding*2), padding_color, dtype='uint8')
    fr = numpy.asarray(captcha.convert('L'))
    bg[padding:padding+height,padding:padding+width] = fr
    
    black_pixel_indexes = numpy.transpose(numpy.nonzero(bg <= 150))
    gmm = sklearn.mixture.GaussianMixture(n_components=7, covariance_type='tied', reg_covar=1e2, tol=1e3, n_init=9)
    gmm.fit(black_pixel_indexes)
        
    indexes = gmm.means_.astype(int).tolist()
    new_indexes = []
    for [y, x] in indexes:
        new_indexes.append((y - padding, x - padding))

    data = numpy.empty((0, 40, 40), 'float32')
    full_image = data_to_image(bg)
    
    for [y, x] in new_indexes:
        cim = full_image.crop((x, y, x + padding*2, y + padding*2))
        X = numpy.asarray(cim.convert('L')).astype('float32')
        X[X <= 150] = -1
        # black
        X[X >  150] = 1
        # white
        data = numpy.append(data, X.reshape(1, 40, 40), axis=0)
        

    return data, new_indexes


def mark_points(image, points):
    '''
    mark locations on image
    '''
    
    im = image.convert("RGB")
    bgdr = ImageDraw.Draw(im)
    for [y, x] in points:
        bgdr.ellipse((x-3, y-3, x+3, y+3), fill ="red", outline ='red')
    return im


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 25, 40)
        self.fc2 = nn.Linear(40, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def data_to_image(d):
    '''
    convert 2darray to image object.
    '''
    
    return Image.fromarray(numpy.uint8(d))


# load net from file.
net = torch.load("./zheye.pt")
net.eval()


def predict_result(filename):
    '''
    given a captcha image file,
    return the upsite down character indexes.
    '''
    
    device = torch.device("cuda")

    data, indexes = CAPTCHA_to_data(filename)
    inputs = torch.from_numpy(data.reshape(7, 1, 40, 40)).to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.tolist()
    
    return [i for (i, p) in zip(indexes, predicted) if not p]


def main(filename):
    ps = predict_result(filename)
    #im = Image.open(filename)
    #mark_points(im, ps)
    print(ps)


import sys
if __name__ == "__main__":
    main(sys.argv[1])
