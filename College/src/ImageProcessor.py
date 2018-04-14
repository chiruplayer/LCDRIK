import os,sys
#import cv2
import glob
import math
import time
import random
import numpy as np
import pandas as pd
import PIL

from PIL import Image
from os import path
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

import lasagne
from lasagne.layers import helper
from lasagne.updates import adam
from lasagne.nonlinearities import rectify, softmax
from lasagne.layers import InputLayer, MaxPool2DLayer, DenseLayer, DropoutLayer, helper
from lasagne.layers import Conv2DLayer as ConvLayer

import theano
from theano import tensor as T

PIXELS = 24
imageSize = PIXELS * PIXELS
num_features = imageSize 
files=[]
newfiles=[]

def load_train_cv(encoder):
    X_train = []
    y_train = []
    for filename in os.listdir('bloatD/'):
        files.append(filename)
    for file in files:
        img = cv2.imread(''+file , 0)
        #img = cv2.resize(img, (PIXELS, PIXELS))
        #img = img.transpose(2, 0, 1)
        img = np.reshape(img, (1, num_features))
        X_train.append(img)
        y_train.append(1)
    for filename in os.listdir(''):
        newfiles.append(filename)
    for file in newfiles:
        img = cv2.imread(''+file , 0)
        #img = cv2.resize(img, (PIXELS, PIXELS))
        #img = img.transpose(2, 0, 1)
        img = np.reshape(img, (1, num_features))
        X_train.append(img)
        y_train.append(0)
    
    X_train = np.array(X_train)
 
    y_train = np.array(y_train)

    y_train = encoder.fit_transform(y_train).astype('int32')

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state = 42)

    X_train = X_train.reshape(X_train.shape[0], 1, PIXELS, PIXELS).astype('float32') / 255.
    X_test = X_test.reshape(X_test.shape[0], 1, PIXELS, PIXELS).astype('float32') / 255.

    return X_train, y_train, X_test, y_test, encoder


#print(files)
#print(newfiles)
def trainer(input_var=None):
    l_in = InputLayer(shape=(None, 1, PIXELS, PIXELS), input_var=input_var)

    l_conv = ConvLayer(l_in, num_filters=8, filter_size=3, pad=1, nonlinearity=rectify)
    l_convb = ConvLayer(l_conv, num_filters=8, filter_size=3, pad=1, nonlinearity=rectify)
    l_pool = MaxPool2DLayer(l_convb, pool_size=2) # feature maps 12x12

    #l_dropout1 = DropoutLayer(l_pool, p=0.25)
    l_hidden = DenseLayer(l_pool, num_units=128, nonlinearity=rectify)
    #l_dropout2 = DropoutLayer(l_hidden, p=0.5)

    l_out = DenseLayer(l_hidden, num_units=10, nonlinearity=softmax)

    return l_out

def iterate(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]

BATCHSIZE = 32
LR = 0.001
ITERS = 5

X = T.tensor4('X')
Y = T.ivector('y')

output_layer = trainer(X)
output_train = lasagne.layers.get_output(output_layer)
output_test = lasagne.layers.get_output(output_layer, deterministic=True)

loss = lasagne.objectives.categorical_crossentropy(output_train, Y)
loss = loss.mean()

valid_loss = lasagne.objectives.categorical_crossentropy(output_test, Y)
valid_loss = valid_loss.mean()

valid_acc = T.mean(T.eq(T.argmax(output_test, axis=1), Y), dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(output_layer, trainable=True)
updates = adam(loss, params, learning_rate=LR)

train_fn = theano.function(inputs=[X,Y], outputs=loss, updates=updates)
valid_fn = theano.function(inputs=[X,Y], outputs=[valid_loss, valid_acc])

predict_proba = theano.function(inputs=[X], outputs=output_test)

encoder = LabelEncoder()
train_X, train_y, valid_X, valid_y, encoder = load_train_cv(encoder)
print('Train shape:', train_X.shape, 'Test shape:', valid_X.shape)

try:
    for epoch in range(ITERS):
        start = time.time()
        train_loss = []
        for batch in iterate(train_X, train_y, BATCHSIZE):
            inputs, targets = batch
            train_loss.append(train_fn(inputs, targets))
        train_loss = np.mean(train_loss)
        valid_loss = []
        valid_acc = []
        for batch in iterate(valid_X, valid_y, BATCHSIZE):
            inputs, targets = batch
            valid_eval = valid_fn(inputs, targets)
            valid_loss.append(valid_eval[0])
            valid_acc.append(valid_eval[1])
        valid_loss = np.mean(valid_loss)
        valid_acc = np.mean(valid_acc)
        ratio = train_loss / valid_loss
        end = time.time() - start
        print('iter:', epoch, '| TL:', np.round(train_loss,decimals=3), '| VL:', np.round(valid_loss, decimals=3), '| Vacc:', np.round(valid_acc, decimals=3), '| Ratio:', np.round(ratio, decimals=2), '| Time:', np.round(end, decimals=1))

except KeyboardInterrupt:
    pass
except KeyboardInterrupt:
    pass


print('Making predictions')
PRED_BATCH = 2
def iterate_pred(inputs, batchsize):
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]

predictions = []
for pred_batch in iterate_pred(valid_X, PRED_BATCH):
    predictions.extend(predict_proba(pred_batch))

predictions = np.array(predictions)

print('pred shape')
print(predictions.shape)

"""
make predictions

print('Making predictions')
PRED_BATCH = 2
def iterate_pred_minibatches(inputs, batchsize):
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]

predictions = []
for pred_batch in iterate_pred_minibatches(X_test, PRED_BATCH):
    predictions.extend(predict_proba(pred_batch))

predictions = np.array(predictions)

print('pred shape')
print(predictions.shape)

print('Creating Submission')
def create_submission(predictions, test_id):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    result1.to_csv('submission_ZFTurboNet.csv', index=False)

create_submission(predictions, X_test_id)
"""



