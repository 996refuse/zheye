#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Recognizing class 

from sklearn.mixture import GaussianMixture
from PIL import Image
from zheye import util
import numpy as np

class zheye:
    def __init__(self):
        ''' load model '''
        import os
        import keras
        full_path = os.path.realpath(__file__)
        path, filename = os.path.split(full_path)
        self.model = keras.models.load_model(path +'/zheyeV3.keras')

    def Recognize(self, fn):
        im = Image.open(fn)
        im = util.CenterExtend(im, radius=20)

        vec = np.asarray(im.convert('L')).copy()
        Y = []
        for i in range(vec.shape[0]):
            for j in range(vec.shape[1]):
                if vec[i][j] <= 200:
                    Y.append([i, j])

        gmm = GaussianMixture(n_components=7, covariance_type='tied', reg_covar=1e2, tol=1e3, n_init=9)
        gmm.fit(Y)
        
        centers = gmm.means_

        points = []
        for i in range(7):
            scoring = 0.0
            for w_i in range(3):
                for w_j in range(3):
                    p_x = centers[i][0] -1 +w_i
                    p_y = centers[i][1] -1 +w_j

                    cr = util.crop(im, p_x, p_y, radius=20)
                    cr = cr.resize((40, 40), Image.ANTIALIAS)

                    X = np.asarray(cr.convert('L'), dtype='float')
                    X = (X.astype("float") - 180) /200

                    x0 = np.expand_dims(X, axis=0)
                    x1 = np.expand_dims(x0, axis=3)

                    global model
                    if self.model.predict(x1)[0][0] < 0.5:
                        scoring += 1

            if scoring > 4:
                points.append((centers[i][0] -20, centers[i][1] -20))
                
        return points