import numpy as np
import pandas as pd
import os
import argparse
import errno
import scipy.misc
import dlib
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

from skimage.feature import hog
if sys.version_info >= (3, 0):
        import _pickle as cPickle
else:
        import cPickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from data_loader import load_data
from parameters import DATASET, TRAINING, HYPERPARAMS

# initialization
image_height = 48
image_width = 48
window_size = 24
window_step = 6

print( "preparing")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_landmarks(image, rects):

    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def sliding_hog_windows(image):
    hog_windows = []

    for y in range(0, image_height, window_step):
        for x in range(0, image_width, window_step):
            window = image[y:y+window_size, x:x+window_size]
            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                            cells_per_block=(1, 1), visualise=False))
    return hog_windows

if os.path.isfile(TRAINING.save_model_path):
    with open(TRAINING.save_model_path, 'rb') as f:
        model = cPickle.load(f)
else:
    print( "Error: file '{}' not found".format(TRAINING.save_model_path))
    exit()

def predict(filename):
    image = cv2.imread(filename,0)
    image = cv2.resize(image,(image_width,image_height),interpolation = cv2.INTER_AREA)

    face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
    face_landmarks = np.array([get_landmarks(image, face_rects)])
    features = face_landmarks

    hog_features = sliding_hog_windows(image) #2592
    hog_features = np.asarray(hog_features)
    face_landmarks = face_landmarks.flatten() #136
    features = np.concatenate((face_landmarks, hog_features))

    predicted_Y = model.predict(features.reshape((1, -1)))

    print " 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral "
    print " answer is:",predicted_Y

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--filename", default=" ", help="the filename you want to predict")
args = parser.parse_args()
predict(args.filename)

