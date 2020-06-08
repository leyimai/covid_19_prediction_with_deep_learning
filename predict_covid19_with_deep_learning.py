#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary package
import os
import numpy as np
from numpy import load
from numpy import savez_compressed
import pandas as pd 
import datetime
import pydot
from IPython.display import SVG
from sklearn.metrics import confusion_matrix
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing import image
from keras import utils
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from keras.layers import SpatialDropout2D
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Model
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
from tensorflow import keras
import seaborn as sn
import scikitplot
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image, ImageOps
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Define the ResNet50 Model
# Reference: https://github.com/AdalbertoCq/Deep-Learning-Specialization-Coursera/blob/master/Convolutional%20Neural%20Networks/week2/ResNet/residual_networks.py

# GRADED FUNCTION: identity_block
def identity_block(X, f, filters, stage, block, dropout_rate=0.2, dropout=False):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    if dropout:
        X = SpatialDropout2D(dropout_rate)(X)
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    if dropout:
        X = SpatialDropout2D(dropout_rate)(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X

# GRADED FUNCTION: convolutional_block
def convolutional_block(X, f, filters, stage, block, s = 2, dropout_rate=0.2, dropout=False):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    if dropout:
        X = SpatialDropout2D(dropout_rate)(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    if dropout:
        X = SpatialDropout2D(dropout_rate)(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    
    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)


    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X


# In[ ]:


# k-fold cross validation
# Reference: https://datascience.stackexchange.com/questions/37009/k-fold-cross-validation-when-using-fit-generator-and-flow-from-directory-in-ke
def kfold(X, Y_train_onehot, n_splits, batch_size, epochs, opt, datagen, augment, class_weight, dropout_rate=0.2, dropout=False):
    kf = KFold(n_splits=n_splits, random_state=None, shuffle=True)
    kf.get_n_splits(X)
    val_accuracy_list = []
    model_list = []
    history_list = []
    i = 1
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_dev = X[test_index]
        y_train = Y_train_onehot[train_index]
        y_dev = Y_train_onehot[test_index]
        print("=========================================")
        print(f"====== {n_splits}-Fold Validation step {i}/{n_splits} =======" )
        print("=========================================")        
        model, history = train_ResNet(X_train, y_train, X_dev, y_dev, batch_size=batch_size, epochs=epochs, opt=opt, datagen=datagen, augment=False, class_weight=None,dropout_rate=dropout_rate, dropout=dropout)
        model_list.append(model)
        history_list.append(history)
        val_accuracy_list.append(history.history['val_accuracy'][-1])
        i+=1
    val_accuracy_array = np.array(val_accuracy_list)
    avg_accuracy= np.mean(val_accuracy_array)
    return  history_list, model_list, avg_accuracy


# In[ ]:


# Define functions for result evaluation
def plot_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(10,10))
    plt.subplot(211)
    y_train = history.history['accuracy']
    y_test = history.history['val_accuracy']
    plt.plot(y_train)
    plt.plot(y_test)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.yticks(np.arange(3,10)/10)
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.figure(figsize=(10,10))
    plt.subplot(211)
    y_train = history.history['loss']
    y_test = history.history['val_loss']
    plt.plot(y_train)
    plt.plot(y_test)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')    
    
def evaluation(model, X, y_true):
    ####### model performance of last epoch#########
    preds = model.evaluate(X, y_true)
    loss = preds[0]
    test_acc = preds[1]
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy =" + str(preds[1]))
    
    ########### model evaluation #####################
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred,axis=1)
    y_true = np.argmax(y_true, axis=1)
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    cm = multilabel_confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)
    print(['bacterial','covid', 'normal','viral'])
    print("confusion matrix: ")
    print(cm)
    print('precision: ', precision)
    print('recall :', recall)
    print('f1 score: ',f1 )
    
    ##### multi-label confusion matrix ##########
    data = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (7,5))
    sn.set(font_scale=1)  #for label size
    ax = sn.heatmap(df_cm, 
                    cmap="Blues", annot=True, fmt='g', 
                    annot_kws={"size": 14})# font size
    ax.set_xticklabels(['bacterial','covid','normal','viral'])
    ax.set_yticklabels(['bacterial','covid','normal','viral'])
    ax.set_ylim(4, -0)
    filename = str(datetime.datetime.now()).replace(" ","").replace(":","").replace("-","").split(".")[0][4:][:-2]
    plt.savefig('multi_confusiom_matrix_' + filename + '_plot.png')
    plt.show()
    
    
    ###### covid confusion matrix #########
    #plt.figure(figsize = (7,5))
    #plt.subplot(221)
    ax1 = sn.heatmap(cm[0], linewidths=.2, cmap="Blues", 
                    xticklabels=['negative','positive'], 
                    yticklabels=['negative','positive'], 
                    annot=True, fmt='g', cbar=True,
                    annot_kws={"size": 14})
    ax1.set_ylim(len(cm[0]), -0.5)
    ax1.title.set_text('Confusion Matrix for Bacterial Pneumonia')
    ax1.set_ylim(2, -0)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.savefig('confusiom_matrix_0_' + filename + '_plot.png')
    plt.show()
    
    #plt.subplot(222)
    ax2 = sn.heatmap(cm[1], linewidths=.2, cmap="Blues", 
                    xticklabels=['negative','positive'], 
                    yticklabels=['negative','positive'], 
                    annot=True, fmt='g', cbar=True,
                    annot_kws={"size": 14})
    ax2.set_ylim(len(cm[1]), -0.5)
    ax2.title.set_text('Confusion Matrix for COVID-19')
    ax2.set_ylim(2, -0)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.savefig('confusiom_matrix_1_' + filename + '_plot.png')
    plt.show()

    #plt.subplot(223)
    ax3 = sn.heatmap(cm[2], linewidths=.2, cmap="Blues", 
                    xticklabels=['negative','positive'], 
                    yticklabels=['negative','positive'], 
                    annot=True, fmt='g', cbar=True,
                    annot_kws={"size": 14})
    ax3.set_ylim(len(cm[2]), -0.5)
    ax3.title.set_text('Confusion Matrix for Healthy')
    ax3.set_ylim(2, -0)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.savefig('confusiom_matrix_2_' + filename + '_plot.png')
    plt.show()
    
    #plt.subplot(224)
    ax4 = sn.heatmap(cm[3], linewidths=.2, cmap="Blues", 
                    xticklabels=['negative','positive'], 
                    yticklabels=['negative','positive'], 
                    annot=True, fmt='g', cbar=True,
                    annot_kws={"size": 14})
    ax4.set_ylim(len(cm[3]), -0.5)
    ax4.title.set_text('Confusion Matrix for Viral Pneumonia')
    ax4.set_ylim(2, -0)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.savefig('confusiom_matrix_3_' + filename + '_plot.png')
    plt.show()
    return loss, test_acc, precision, recall, f1, cm


# In[ ]:


# Define function for prediction
def make_prediction(model, X_test):
    y_predict = model.predict(X_test)
    y_predict = np.argmax(y_predict,axis=1)
    inv_label_dict = {value:key for key,value in label_dict.items()}
    output = pd.DataFrame(zip(test_names,y_predict),columns=['filename','predict']) 
    output['filename'] = output['filename'].apply(lambda x: f'test/{x}')
    output['label'] = output['predict'].apply(lambda x: inv_label_dict.get(x))
    submission = pd.merge(test, output, left_on='filename',right_on='filename').drop(columns=['predict','filename'])
    cur_time = str(datetime.datetime.now()).replace(" ","").replace(":","").replace("-","").split(".")[0][4:][:-2]
    submission.to_csv('submission_'+cur_time+'.csv',index=False)


# In[ ]:


# img_path = '/kaggle/input/4771-sp20-covid/train/train/img-132.jpeg'
def resize_with_pad(img_array, img_size=224):
    #im = Image.open(im_pth)
    im = Image.fromarray(img_array)
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (img_size, img_size))
    new_im.paste(im, ((img_size-new_size[0])//2,
                        (img_size-new_size[1])//2))
    return np.array(new_im)


# # 1. Data Preprocessing

# In[ ]:


# # import and preprocess images (resize with padding)
# train = pd.read_csv('../input/4771-sp20-covid/train.csv')
# test = pd.read_csv('../input/4771-sp20-covid/test.csv')
# print(train.groupby("label").size())

# # enhance contrast with CLAHE
# clahe = cv.createCLAHE(clipLimit=10, tileGridSize=(8,8))

# #read training images
# img_size=224
# X  = []
# Y = []
# for dirname, _, filenames in os.walk('/kaggle/input/4771-sp20-covid/train/train'):
#     for filename in filenames:
#         img_path = os.path.join(dirname, filename)
#         img = cv.imread(img_path,0)        # use grayscale to input into clahe
#         x = clahe.apply(img)
#         x = cv.cvtColor(x,cv.COLOR_GRAY2RGB)   # transform grayscale to RGB
#         x = resize_with_pad(x, img_size=img_size)
#         X.append(x)
#         y = list(train[train.filename=='train/'+filename].label)
#         Y+=y 
# X = np.array(X)
# Y = np.array(Y)

# #read testing images
# X_test = []
# test_names = []
# for dirname, _, filenames in os.walk('/kaggle/input/4771-sp20-covid/test/test'):
#     for filename in filenames:
#         img_path = os.path.join(dirname, filename)
#         img = cv.imread(img_path,0)        # use grayscale to input into clahe
#         x = clahe.apply(img)
#         x = cv.cvtColor(x,cv.COLOR_GRAY2RGB)   # transform grayscale to RGB        
#         x = resize_with_pad(x, img_size=img_size)
#         X_test.append(x)
#         test_names.append(filename)
# X_test = np.array(X_test)
# test_names = np.array(test_names)

# # data quality check
# print(X.shape)
# print(Y.shape)
# print(X_test.shape)
# print(test_names.shape)
# plt.subplot(121)
# plt.imshow(X[0])
# plt.subplot(122)
# plt.imshow(X_test[0])
# plt.show()

# # # Normalize the image vectors from (0,255) to (0,1)
# # print(np.max(X), np.min(X))
# # print(np.max(X_test), np.min(X_test))
# # X = X/255
# # X_test = X_test/255

# # Transform text labels into one-hot labels
# label_dict = {'bacterial':0,'covid':1, 'normal':2,'viral':3}   
# inv_label_dict = {value:key for key,value in label_dict.items()}
# Y_train = np.array([label_dict.get(i) for i in Y] )
# Y_train_onehot = to_categorical(Y_train,4)

# # train/validation split
# X_train, X_dev, y_train, y_dev = train_test_split(X, Y_train_onehot, test_size=0.1, random_state=2) 
# print ("X_train shape: " + str(X_train.shape))
# print ("y_train shape: " + str(y_train.shape))
# print ("X_test shape: " + str(X_dev.shape))
# print ("y_test shape: " + str(y_dev.shape))
# values, cnts = np.unique(np.argmax(y_dev,axis=1), return_counts=True)
# print(list(zip(inv_label_dict.values(), cnts)))


# In[ ]:


# import and preprocess images (resize without padding)
train = pd.read_csv('../input/4771-sp20-covid/train.csv')
test = pd.read_csv('../input/4771-sp20-covid/test.csv')
print(train.groupby("label").size())

# enhance contrast with CLAHE
clahe = cv.createCLAHE(clipLimit=10, tileGridSize=(8,8))

#read training images
img_size=224
X  = []
Y = []
for dirname, _, filenames in os.walk('/kaggle/input/4771-sp20-covid/train/train'):
    for filename in filenames:
        img_path = os.path.join(dirname, filename)
        img = cv.imread(img_path,0)        # use grayscale to input into clahe
        x = clahe.apply(img)
        x = cv.cvtColor(x,cv.COLOR_GRAY2RGB)   # transform grayscale to RGB
        x = cv.resize(x, dsize=(img_size, img_size), interpolation=cv.INTER_LINEAR)   #resize image
        X.append(x)
        y = list(train[train.filename=='train/'+filename].label)
        Y+=y 
X = np.array(X)
Y = np.array(Y)

#read testing images
X_test = []
test_names = []
for dirname, _, filenames in os.walk('/kaggle/input/4771-sp20-covid/test/test'):
    for filename in filenames:
        img_path = os.path.join(dirname, filename)
        img = cv.imread(img_path,0)         # use grayscale to input into clahe
        x = clahe.apply(img)
        x = cv.cvtColor(x,cv.COLOR_GRAY2RGB)   # transform grayscale to RGB
        x = cv.resize(x, dsize=(img_size, img_size), interpolation=cv.INTER_LINEAR)   #resize image
        X_test.append(x)
        test_names.append(filename)
X_test = np.array(X_test)
test_names = np.array(test_names)

# data quality check
print(X.shape)
print(Y.shape)
print(X_test.shape)
print(test_names.shape)
plt.subplot(121)
plt.imshow(X[0])
plt.subplot(122)
plt.imshow(X_test[0])
plt.show()

# # Normalize the image vectors from (0,255) to (0,1)
# print(np.max(X), np.min(X))
# print(np.max(X_test), np.min(X_test))
# X = X/255
# X_test = X_test/255

# Transform text labels into one-hot labels
label_dict = {'bacterial':0,'covid':1, 'normal':2,'viral':3}   
inv_label_dict = {value:key for key,value in label_dict.items()}
Y_train = np.array([label_dict.get(i) for i in Y] )
Y_train_onehot = to_categorical(Y_train,4)

# train/validation split
X_train, X_dev, y_train, y_dev = train_test_split(X, Y_train_onehot, test_size=0.1, random_state=27) 
print ("X_train shape: " + str(X_train.shape))
print ("y_train shape: " + str(y_train.shape))
print ("X_test shape: " + str(X_dev.shape))
print ("y_test shape: " + str(y_dev.shape))
values, cnts = np.unique(np.argmax(y_dev,axis=1), return_counts=True)
print(list(zip(inv_label_dict.values(), cnts)))


# # 2. Model Training

# In[ ]:


# GRADED FUNCTION: ResNet50
def ResNet50(input_shape = (64, 64, 3), classes = 4, dropout_rate=0.2, last_dropout_rate=0.5, dropout=False):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    if dropout:
        X = SpatialDropout2D(dropout_rate)(X)
    
    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1, dropout_rate=dropout_rate, dropout=dropout)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b',dropout_rate=dropout_rate, dropout=dropout)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c',dropout_rate=dropout_rate, dropout=dropout)
    if dropout:
        X = SpatialDropout2D(dropout_rate)(X)
    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block='a', s = 2, dropout_rate=0.2, dropout=False)
    X = identity_block(X, 3, [128,128,512], stage=3, block='b',dropout_rate=dropout_rate, dropout=dropout)
    X = identity_block(X, 3, [128,128,512], stage=3, block='c',dropout_rate=dropout_rate, dropout=dropout)
    X = identity_block(X, 3, [128,128,512], stage=3, block='d',dropout_rate=dropout_rate, dropout=dropout)
    if dropout:
        X = SpatialDropout2D(dropout_rate)(X)

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2,dropout_rate=0.2, dropout=False)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b',dropout_rate=dropout_rate, dropout=dropout)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c',dropout_rate=dropout_rate, dropout=dropout)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d',dropout_rate=dropout_rate, dropout=dropout)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e',dropout_rate=dropout_rate, dropout=dropout)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f',dropout_rate=dropout_rate, dropout=dropout)
    if dropout:
        X = SpatialDropout2D(dropout_rate)(X)

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2,dropout_rate=0.2, dropout=False)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b',dropout_rate=dropout_rate, dropout=dropout)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c',dropout_rate=dropout_rate, dropout=dropout)
    if dropout:
        X = SpatialDropout2D(dropout_rate)(X)

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2,2))(X)
    
    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    #X = Dense(128,  activation='relu')(X)   #add a dense layer
    #X = SpatialDropout2D(last_dropout_rate)(X)            #spatial dropout 50% of the neurons
    X = Dropout(last_dropout_rate)(X)            #spatial dropout 50% of the neurons
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


# In[ ]:


# print(f"training accuracy: {history.history['accuracy'][-1]:.3f} , validation accuracy: {history.history['val_accuracy'][-1]:.3f}")
# print(f"training loss: {history.history['loss'][-1]:.3f} , validation loss: {history.history['val_loss'][-1]:.3f}")


# In[ ]:


# # weights_file = 'weights-improvement-02-0.28.hdf5'
# weights_file = "weights.best.hdf5"
# model_opt = opt_model(weights_file=weights_file, opt=opt, dropout_rate=dropout_rate, last_dropout_rate=last_dropout_rate, dropout=dropout)
# make_prediction(model_opt, X_test)


# #### new format

# In[ ]:


def resnet50(weights=None, opt='adam', last_dropout_rate=0.5):
    model = ResNet50(input_shape = (img_size, img_size, 3), classes =4, last_dropout_rate=last_dropout_rate)
    model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
    if weights:
        model.load_weights(weights)
    return model


# In[ ]:


# set up hyperparameters
checkpoint = ModelCheckpoint("resnet50.weights.best.hdf5", monitor='val_loss', save_best_only=True, mode='min')  #verbose=1
callbacks_list = [checkpoint]
datagen = ImageDataGenerator(
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=True,
                            brightness_range=(0.5,1.5),
                            zoom_range=0.2
                            )

batch_size=32
epochs = 200
# opt='adam'
# opt = keras.optimizers.Adam(learning_rate=1e-3)
opt = keras.optimizers.Adam(learning_rate=0.0002)
# class_weight = {0:0.25, 1:0.3, 2:0.2, 3:0.25} #{0 : 0.25, 1 : 0.4, 2 : 0.1, 3 : 0.25}
class_weight=None
last_dropout_rate=0.5


# In[ ]:


resnet50_model = resnet50(opt=opt,last_dropout_rate=last_dropout_rate)
resnet50_history = resnet50_model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), 
                              epochs=epochs,
                              shuffle=True,
                              validation_data=(X_dev, y_dev),
                              class_weight=class_weight,
                              callbacks=callbacks_list,
                              verbose=1) 
plot_history(resnet50_history)


# In[ ]:


opt_resnet50_model = resnet50(weights="resnet50.weights.best.hdf5", opt=opt, last_dropout_rate=last_dropout_rate)
make_prediction(opt_resnet50_model, X_test)

