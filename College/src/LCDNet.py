import os,sys
import cv2
import glob
import math
import time
import random
import numpy as np
import pandas as pd
import PIL
import matplotlib.pyplot as plt
import csv

from PIL import Image
from os import path
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

import lasagne
from lasagne.layers import helper
from lasagne.updates import adam
from lasagne.nonlinearities import rectify, softmax, very_leaky_rectify, sigmoid, elu, tanh
from lasagne.layers import InputLayer, MaxPool2DLayer, DenseLayer, DropoutLayer, helper
from lasagne.layers import Conv2DLayer as ConvLayer

import theano
from theano import tensor as T

PIXELS = 24
imageSize = PIXELS * PIXELS
num_features = imageSize 
files=[]
newfiles=[]
testfiles = []

def load_train_cv(encoder):
    X_train = []
    y_train = []
    print('Read train images')
    for filename in os.listdir('/Users/lorjain/Downloads/bloatH'):
        files.append(filename)
    for file in files:
        img = cv2.imread('/Users/lorjain/Downloads/bloatH/'+file , 0)
        img = cv2.resize(img, (PIXELS, PIXELS))
        #img = img.transpose(2, 0, 1)
        img = np.reshape(img, (1, num_features))
        X_train.append(img)
        y_train.append(1)
    for filename in os.listdir('/Users/lorjain/Downloads/bloat'):
        newfiles.append(filename)
    for file in newfiles:
        img = cv2.imread('/Users/lorjain/Downloads/bloat/'+file , 0)
        img = cv2.resize(img, (PIXELS, PIXELS))
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

def load_test():
    print("Read test images")
    X_test = []
    X_test_id =[]
    testfiles = []
    for filename in os.listdir('/Users/lorjain/Downloads/bloatD'):
        testfiles.append(filename)
    for file in testfiles:
        fl = os.path.basename(file)
        img = cv2.imread('/Users/lorjain/Downloads/bloatD/'+file , 0)
        img = cv2.resize(img, (PIXELS, PIXELS))
        #img = img.transpose(2, 0, 1)
        img = np.reshape(img, (1, num_features))
        X_test.append(img)
        X_test_id.append(fl)
        
    X_test = np.array(X_test)
    X_test_id = np.array(X_test_id)

    X_test = X_test.reshape(X_test.shape[0], 1, PIXELS, PIXELS).astype('float32') / 255.

    return X_test, X_test_id

#print(files)
#print(newfiles)
'''
Lasagne Model LCD_NET and Batch Iterator
'''
def LCD_NET(input_var=None):
    l_in = InputLayer(shape=(None, 1, PIXELS, PIXELS), input_var=input_var)

    l_conv = ConvLayer(l_in, num_filters=3, filter_size=3, pad=1, nonlinearity=very_leaky_rectify)
    l_convb = ConvLayer(l_conv, num_filters=3, filter_size=3, pad=1, nonlinearity=very_leaky_rectify)
    #l_conv1 = ConvLayer(l_convb, num_filters=8, filter_size=3, pad=1, nonlinearity=rectify)
    #l_conv2 = ConvLayer(l_conv1, num_filters=8, filter_size=3, pad=1, nonlinearity=rectify)

    l_pool = MaxPool2DLayer(l_convb, pool_size=2) # feature maps 12x12

    #l_dropout1 = DropoutLayer(l_pool, p=0.25)
    l_hidden = DenseLayer(l_pool, num_units=128, nonlinearity=very_leaky_rectify)
    #l_dropout2 = DropoutLayer(l_hidden, p=0.5)
        
    l_hidden1 = DenseLayer(l_hidden, num_units=64, nonlinearity=very_leaky_rectify)

    l_out = DenseLayer(l_hidden1, num_units=2, nonlinearity=softmax)

    return l_out

def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]

"""
Set up all theano functions
"""
#def labels():
    #pass
BATCHSIZE = 32
LR = 0.001
ITERS = 5

X = T.tensor4('X')
Y = T.ivector('y')

# set up theano functions to generate output by feeding data through network, any test outputs should be deterministic
output_layer = LCD_NET(X)
#print(output_layer)
output_train = lasagne.layers.get_output(output_layer)
#print(output_train)
output_test = lasagne.layers.get_output(output_layer, deterministic=True)

# set up the loss that we aim to minimize, when using cat cross entropy our Y should be ints not one-hot
loss = lasagne.objectives.categorical_crossentropy(output_train, Y)
loss = loss.mean()

# set up loss functions for validation dataset
valid_loss = lasagne.objectives.categorical_crossentropy(output_test, Y)
valid_loss = valid_loss.mean()

valid_acc = T.mean(T.eq(T.argmax(output_test, axis=1), Y), dtype=theano.config.floatX)

# get parameters from network and set up sgd with nesterov momentum to update parameters
params = lasagne.layers.get_all_params(output_layer, trainable=True)
updates = adam(loss, params, learning_rate=LR)

# set up training and prediction functions
train_fn = theano.function(inputs=[X,Y], outputs=loss, updates=updates)
valid_fn = theano.function(inputs=[X,Y], outputs=[valid_loss, valid_acc])

# set up prediction function
predict_proba = theano.function(inputs=[X], outputs=output_test)

#load training data.
encoder = LabelEncoder()
train_X, train_y, valid_X, valid_y, encoder = load_train_cv(encoder)
print('Train shape:', train_X.shape, 'Test shape:', valid_X.shape)

#load test data
X_test, X_test_id = load_test()

# loop over training functions for however many iterations, print information while training
try:
    for epoch in range(ITERS):
        # do the training
        start = time.time()
        # training batches
        train_loss = []
        for batch in iterate_minibatches(train_X, train_y, BATCHSIZE):
            inputs, targets = batch
            train_loss.append(train_fn(inputs, targets))
        train_loss = np.mean(train_loss)
        # validation batches
        valid_loss = []
        valid_acc = []
        for batch in iterate_minibatches(valid_X, valid_y, BATCHSIZE):
            inputs, targets = batch
            valid_eval = valid_fn(inputs, targets)
            valid_loss.append(valid_eval[0])
            valid_acc.append(valid_eval[1])
        valid_loss = np.mean(valid_loss)
        valid_acc = np.mean(valid_acc)
        # get ratio of TL to VL
        ratio = train_loss / valid_loss
        end = time.time() - start
        # print training details
        print('iteration:', epoch, '| Train Loss:', np.round(train_loss,decimals=3), '| Validation Loss:', np.round(valid_loss, decimals=3), '| Validation accuracy(%):', np.round(valid_acc, decimals=3)*100, '| Ratio:', np.round(ratio, decimals=2), '| Time:', np.round(end, decimals=1))

except KeyboardInterrupt:
    pass
except KeyboardInterrupt:
    pass


#make predictions
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
    pass
    result1 = pd.DataFrame(predictions, columns=['Diseased', 'Healthy'])
    result1.loc[:, 'Test Images'] = pd.Series(test_id, index=result1.index)
    result1.to_csv('/Users/lorjain/Downloads/sub.csv', index=False)
    '''
    columns = 2
    rows = 100
    with open("/home/lorick/Desktop/sub.csv") as outfile:
        for x in range(rows):
            a_list = [random.randint(0,1) for i in range(columns)]                                                    
            values = ",".join(str(i) for i in a_list)
            print (values)
            #outfile.write(values + "\n")
    '''
create_submission(predictions, X_test_id)
#labels()
data = list(csv.reader(open('/Users/lorjain/Downloads/sub.csv', 'r')))
#ndata = np.array(data).astype(np.float)
#print(data)
data = data[2:]
ndata = np.array(data)
ndata = ndata[:,:2].astype(np.float)
#print(ndata)
diseased = list(ndata[:,0])
healthy = list(ndata[:,1])
num_of_diseased = len(diseased)
num_of_healthy = len(healthy)
tp = 0
fn = 0
tn = 0
fp = 0
for i in diseased:
    if i < 0.5:
        fp += 1
for i in healthy:
    if i > 0.0015:
        fp += 1
tn = len(diseased) - fp
#precision = tp / (tp + fp)
#recall = tp / (tp + fn)
accuracy = valid_acc
confusion_matrix  = np.array([tp, fn, tn, fp]).reshape((2,2))
print(confusion_matrix)



