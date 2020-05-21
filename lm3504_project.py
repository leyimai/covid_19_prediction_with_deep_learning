#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
from numpy import load
from numpy import savez_compressed
import pandas as pd 
import datetime
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras import utils
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from sklearn.metrics import confusion_matrix
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
from tqdm import tqdm
from tensorflow import keras
import seaborn as sn
import scikitplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Define functions for reading and loading data
def read_an_img(filename):
    img_path = '../input/4771-sp20-covid/train/' + filename
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img = image.img_to_array(img)
    img = img/255
    return img

def load_data(img_size=128):   #One-time data loading
    train = pd.read_csv('../input/4771-sp20-covid/train.csv')
    test = pd.read_csv('../input/4771-sp20-covid/test.csv')
    print(train.groupby("label").size())
    #read training images
    X  = []
    Y = []
    for dirname, _, filenames in os.walk('/kaggle/input/4771-sp20-covid/train/train'):
        for filename in filenames:
            img_path = os.path.join(dirname, filename)
            img = image.load_img(img_path, target_size=(img_size, img_size))
                                 #,color_mode="grayscale")
            x = image.img_to_array(img)
            X.append(x)
            y = list(train[train.filename==f'train/{filename}'].label)
            Y+=y 
    savez_compressed('train_224.npz', X, Y)
    #read testing images
    X_test = []
    test_names = []
    for dirname, _, filenames in os.walk('/kaggle/input/4771-sp20-covid/test/test'):
        for filename in filenames:
            img_path = os.path.join(dirname, filename)
            img = image.load_img(img_path, target_size=(img_size, img_size))
                                 #, color_mode="grayscale")
            x = image.img_to_array(img)
            X_test.append(x) 
            test_names.append(filename)
    savez_compressed('test_224.npz', X_test, test_names)


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

# GRADED FUNCTION: ResNet50
def ResNet50(input_shape = (64, 64, 3), classes = 6, dropout_rate=0.2, dropout=False):
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
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


# In[ ]:


# Build functions for fine tuning
def train_ResNet(X_train, y_train, X_dev, y_dev, datagen, batch_size=16, epochs=3, opt='adam', augment=True, class_weight=None, dropout_rate=0.2, dropout=False ):
    model = ResNet50(input_shape = (img_size, img_size, 3), classes =4, dropout_rate=dropout_rate, dropout=dropout)
    model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy']) 
    if augment:
        history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), 
                                      epochs=epochs,
                                      shuffle=True,
                                      validation_data=(X_dev, y_dev),
                                      class_weight=class_weight,
                                      verbose=1)
    else:
        history = model.fit(X_train, y_train, batch_size=batch_size,
                            epochs=epochs,
                            shuffle=True,
                            validation_data=(X_dev, y_dev),
                            class_weight=class_weight,
                            verbose=1)
        
    return model, history

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


# # 1. Data Preprocessing

# In[ ]:


# Import and preprocess data
train = pd.read_csv('../input/4771-sp20-covid/train.csv')
test = pd.read_csv('../input/4771-sp20-covid/test.csv')

img_size=224
load_data(img_size)  # This will save output to numpy files
train_npz = load('train_224.npz')
X, Y = train_npz['arr_0'], train_npz['arr_1']
test_npz = load('test_224.npz')
X_test, test_names = test_npz['arr_0'], test_npz['arr_1']
label_dict = {'bacterial':0,'covid':1, 'normal':2,'viral':3}   
inv_label_dict = {value:key for key,value in label_dict.items()}
Y_train = np.array([label_dict.get(i) for i in Y] )
Y_train_onehot = to_categorical(Y_train,4)
X_train_orig, X_dev_orig, y_train, y_dev = train_test_split(X, Y_train_onehot, test_size=0.1, random_state=42) 

# Normalize the image vectors from (0,255) to (0,1)
X_train = X_train_orig/255.
X_dev = X_dev_orig/255.

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_test shape: " + str(X_dev.shape))
print ("Y_test shape: " + str(y_dev.shape))


# # 2. Model Training

# In[ ]:


# Model 1: Base Model 

# Tunining batch size
batch_size_list = [8,16,32]
epochs_list = [30,40,50,60]
score = []        #to store training acc
model_history_list = []
for batch_size in tqdm(batch_size_list):
    model, history = train_ResNet(X_train, y_train, X_dev, y_dev, datagen=False, batch_size=batch_size, epochs=60, opt='adam', augment=False, class_weight=None)
    model_history_list.append((model,history))
    acc_test = history.history['val_accuracy'][-1]
    #acc_train = model.evaluate(X, Y_train_onehot)[1]
    #acc_test = model.evaluate(X_dev, y_dev)[1]
    score.append( {"batch_size": batch_size, "final accuracy on validation data": acc_test} )

# visualize the results
history_array = np.array([tup[1] for tup in model_history_list])
for history in history_array:
    plot_history(history)
val_acc_batch_8 = np.array([history.history['val_accuracy'] for history in history_array][0])
print("batchsize=8 ", "val_accuracy ", np.argmax(val_acc_batch_8), np.max(val_acc_batch_8))
acc_batch_8 = np.array([history.history['accuracy'] for history in history_array][0])
print("batchsize=8 ","overall accuracy ", np.argmax(acc_batch_8), np.max(acc_batch_8))
val_acc_batch_16 = np.array([history.history['val_accuracy'] for history in history_array][1])
print("batchsize=16 ","val_accuracy ", np.argmax(val_acc_batch_16), np.max(val_acc_batch_16))
acc_batch_16 = np.array([history.history['accuracy'] for history in history_array][1])
print("batchsize=16 ","overall accuracy ",np.argmax(acc_batch_16), np.max(acc_batch_16))
val_acc_batch_32 = np.array([history.history['val_accuracy'] for history in history_array][2])
print("batchsize=32 ","val_accuracy ", np.argmax(val_acc_batch_32), np.max(val_acc_batch_32))
acc_batch_32 = np.array([history.history['accuracy'] for history in history_array][2])
print("batchsize=32 ","overall accuracy ",np.argmax(acc_batch_32), np.max(acc_batch_32))

model_array = np.array([tup[0] for tup in model_history_list])
i = 1
for model in model_array:
    print("==================================================================")
    print(f"============== model {i} ：batch size={batch_size_list[i-1]}=====================")
    print("==================================================================")
    print()
    print("performance on holdout test data")
    loss, test_acc, precision, recall, f1, confusion_matrix = evaluation(model, X_dev, y_dev)
    print()
    print("performance on all training data")
    loss, test_acc, precision, recall, f1, confusion_matrix = evaluation(model, X, Y_train_onehot)
    i += 1
    
# Pick optimal batch size as 32


# In[ ]:


# 10-fold cross validation for tuning
n_splits=10
epochs = 80
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1)
opt = 'Adam'
# opt = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

batch_size_list = [16, 32]
augment_list = [True, False]
class_weight_list = [None, {0:0.25, 1:0.3, 2:0.2, 3:0.25}]  #{0 : 0.25, 1 : 0.4, 2 : 0.1, 3 : 0.25}
dropout_rate_list = [0.2, 0.4]

result = []
for batch_size in batch_size_list:
    for augment in augment_list:
        for class_weight in class_weight_list:
            for dropout_rate in dropout_rate_list:
                history_list, model_list, avg_accuracy = kfold(X, Y_train_onehot, n_splits, batch_size, epochs, opt, datagen, augment, class_weight, dropout_rate, dropout=True)
                result.append({"batch_size":batch_size, "augment":augment, "class_weight": class_weight, "dropout_rate": dropout_rate, "history": history_list, "model":model_list, "accuracy": avg_accuracy})


# In[ ]:


# Model 2: Base Model + dropout (droptout_rate=0.2)
epochs = 100
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1)
opt = 'Adam'
model, history = train_ResNet(X_train, y_train, X_dev, y_dev, datagen, batch_size=32, epochs=epochs, opt=opt, augment=True, class_weight=None, dropout_rate=0.2, dropout=True )


# In[ ]:


plot_history(history)


# Adding dropout could improve and stablize performance by prevent overfitting.

# In[ ]:


make_prediction(model, X_test)


# In[ ]:


# Model 3: Base Model + dropout + data augmentation

epochs = 60
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1)
opt = 'Adam'
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

augment_list = [True, False]
class_weight = {0:0.25, 1:0.3, 2:0.2, 3:0.25}   #{0 : 0.25, 1 : 0.4, 2 : 0.1, 3 : 0.25}
dropout_rate_list = [0.2, 0.4]

model, history = train_ResNet(X_train, y_train, X_dev, y_dev, datagen, batch_size=32, epochs=epochs, opt='adam', augment=True, class_weight=class_weight, dropout_rate=0.2, dropout=True )


# In[ ]:


plot_history(history)


# Using data augmentation further mitigate overfitting.

# In[ ]:


make_prediction(model, X_test)


# In[ ]:


# Model 4: Base Model + dropout + data augmentation + self-defined Learning Rate (0.0005)
epochs = 100
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1) 
opt = keras.optimizers.Adam(learning_rate=0.0005)

augment_list = [True, False]
# class_weight = {0:0.25, 1:0.3, 2:0.2, 3:0.25} #{0 : 0.25, 1 : 0.4, 2 : 0.1, 3 : 0.25}
dropout_rate_list = [0.2, 0.4]

model, history = train_ResNet(X_train, y_train, X_dev, y_dev, datagen, batch_size=32, epochs=epochs, opt=opt, augment=True, class_weight=None, dropout_rate=0.2, dropout=True )


# Model 4 yields the best overall performance by slowering down the learning rate.

# In[ ]:


make_prediction(model, X_test)


# In[ ]:


plot_history(history)


# In[ ]:


model.save('final_model.tf')
model_1 = load_model('final_model.tf')


# # 3. Evaluation on Best Model (Model 4)

# In[ ]:


# metrics on the total training set
loss, test_acc, precision, recall, f1, cm = evaluation(model, X, Y_train_onehot)


# In[ ]:


# metrics on the 10% testing set
loss, test_acc, precision, recall, f1, cm = evaluation(model, X_dev, y_dev)


# In[ ]:


# plot ROC curve and precision-recall curve
# Reference: https://scikit-plot.readthedocs.io/en/stable/metrics.html
y_true = np.argmax(y_dev, axis=1)
y_label = np.array([inv_label_dict.get(i) for i in y_true])
y_probas = model.predict(X_dev)       
scikitplot.metrics.plot_roc(y_label, y_probas, plot_micro=True, plot_macro=False, figsize=(10,8), text_fontsize=10)
scikitplot.metrics.plot_precision_recall(y_label, y_probas,figsize=(10,8), text_fontsize=10)
plt.show()


# # 5. Visualization and interpretation

# In[ ]:


# plot model
plot_model(model, to_file='ResNet.png')
keras.optimizers.Adam(model_to_dot(model).create(prog='dot'))
print(model.summary())


# In[ ]:


# Plot the X-rays samples for each class

# Select 2 samples for each class
img_samples=[]
i = 0
while i<2:
    for key in label_dict.keys():
        file = list(train[train['label'] == key].sample(1)['filename'])[0]
        img = read_an_img(file)
        img_samples.append(img)
    i+=1
print(len(img_samples))

# plot these sample X-rays
fig = plt.figure(figsize=(15,8))
fig.subplots_adjust(hspace=0.0001, wspace=0.2)
for i in range(8):
    ax = fig.add_subplot(2, 4, i+1)
    ax = plt.imshow(img_samples[i])
    plt.xticks([]),plt.yticks([])
    row, col = i//4, i%4
    if row == 0:
        title = inv_label_dict.get(col)
        plt.title(title,fontsize=18)
plt.show()


# In[ ]:


# Interpret output of 
# Reference: https://www.analyticsvidhya.com/blog/2019/05/understanding-visualizing-neural-networks/

#defining names of layers from which we will take the output
layer_names = ['res2c_branch2c','res3d_branch2c','res4f_branch2c','res5c_branch2c']
outputs = []
img = read_an_img(filename)
img = img.reshape(1, 224, 224, 3)  # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

#extracting the output and appending to outputs
for layer_name in layer_names:
    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(img)
    outputs.append(intermediate_output)
#plotting the outputs
fig,ax = plt.subplots(nrows=4,ncols=5,figsize=(10,10))

for i in range(4):
    for z in range(5):
        ax[i][z].imshow(outputs[i][0,:,:,z])
        ax[i][z].set_title(layer_names[i])
        ax[i][z].set_xticks([])
        ax[i][z].set_yticks([])
plt.savefig('layerwise_output_2.jpg')


# ### with LIME

# In[ ]:


def single_predict(file):
    img = read_an_img(file)
    img = img.reshape(1,img_size,img_size,3)
    probs = model.predict(img)
    indx = np.argmax(probs)
    return inv_label_dict.get(indx)


# In[ ]:


train['predict'] = train['filename'].apply(lambda x: single_predict(x))


# In[ ]:


train.groupby('predict').size()


# In[ ]:


# Reference: https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Image%20Classification%20Keras.ipynb
import os
import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
print('Notebook run using keras:', keras.__version__)


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..'))   # add the current directory
    import lime
from lime import lime_image


# In[ ]:


explainer = lime_image.LimeImageExplainer()


# ### Explanation for prediction of 'bacterial'

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Hide color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels\nexplanation = explainer.explain_instance(img_samples[0], model.predict, top_labels=5, hide_color=0, num_samples=1000)')


# In[ ]:


from skimage.segmentation import mark_boundaries
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False, min_weight=0.01)
print(explanation.top_labels[0])
plt.figure(figsize=(6,6))
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.xticks([]), plt.yticks([])
plt.savefig('feature_importance_bacterial' + '_plot.png')
plt.show()


# ### Explanation for prediction of 'covid'

# In[ ]:


file = train[(train.label=='covid') & (train.predict=='covid')].iloc[0]['filename']
img = read_an_img(file)

plt.figure(figsize=(8,8))
plt.imshow(img)
plt.xticks([]),plt.yticks([])
plt.show()

# Hide color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels
explanation = explainer.explain_instance(img, model.predict, top_labels=5, hide_color=0, num_samples=1000)
from skimage.segmentation import mark_boundaries
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False, min_weight=0.01)
print(explanation.top_labels[0])
plt.figure(figsize=(6,6))
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.xticks([]), plt.yticks([])
plt.savefig('feature_importance_bacterial' + '_plot.png')
plt.show()


# ### Explanation for prediction of 'viral'

# In[ ]:


file = train[(train.label=='viral') & (train.predict=='viral')].iloc[0]['filename']
img = read_an_img(file)

plt.figure(figsize=(8,8))
plt.imshow(img)
plt.xticks([]),plt.yticks([])
plt.show()

# Hide color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels
explanation = explainer.explain_instance(img, model.predict, top_labels=5, hide_color=0, num_samples=1000)
from skimage.segmentation import mark_boundaries
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False, min_weight=0.01)
print(explanation.top_labels[0])
plt.figure(figsize=(6,6))
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.xticks([]), plt.yticks([])
plt.savefig('feature_importance_bacterial' + '_plot.png')
plt.show()

