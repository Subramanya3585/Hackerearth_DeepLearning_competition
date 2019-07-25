# importing required library

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from glob import glob


# reading training and testing data
train =  pd.read_csv("meta-data/train.csv")
test = pd.read_csv('meta-data/test.csv')

train.head()

col = list(train.columns)

Train_Path = 'train/'
Test_Path = 'test/'

#importing opencv and looking at few images

from PIL import Image
import cv2

img_path = Train_Path+str(train.Image_id[1])
img_path_test = Test_Path+str(test.Image_id[1])

Image.open(img_path)

img = cv2.imread(img_path)
img.shape

label_cols = list(set(train.columns)-set(['Image_id']))
label_cols.sort()

labels = train.iloc[1][2:].index[train.iloc[1][2:]==0]
txt = 'Labels/ Attributes: ' + str(labels.values)
ax = plt.figure(figsize=(10, 10))
ax.text(.5, .05, txt, ha='center')
plt.imshow(img)

#data preprocessing

from tqdm import tqdm

def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128,128))
    return img


train_img = []
for img_path in tqdm(train.Image_id.values):
    train_img.append(read_img(Train_Path + img_path))

import gc

#convert image into array for train
X_train = np.array(train_img,np.float32)/255

del train_img
gc.collect()

mean_img = X_train.mean(axis= 0)
std_img =  X_train.std(axis=0)
X_norm = (X_train - mean_img)/std_img

X_norm.shape
del X_train
gc.collect()

y = train[label_cols].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y1 = le.fit_transform(y)

# spliting into train and vaild set
y2 = y1.ravel()
y2 = np.array(y2).astype(int)

from sklearn.model_selection import train_test_split

Xtrain, Xvalid, Ytrain, Yvaild = train_test_split(X_norm, y2, test_size = 0.3, random_state = 123) 

del X_norm
gc.collect()

#preparing data for the test dataset
test_img = []
for img_path_test in tqdm(test.Image_id.values):
    test_img.append(read_img(Test_Path+ img_path_test))

#converting into array for test

X_predd = np.array(test_img, np.float32)/255    

del test_img
gc.collect()

mean_img_test = X_predd.mean(axis = 0)
std_img_test = X_predd.std(axis= 0)
X_norm_test = (X_predd - mean_img_test)/std_img_test

X_norm_test.shape
del X_predd
gc.collect()    


#building the model

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
#from keras.layers import advanced_activations, LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
#
import dill
filename = 'dump_file/globalsave1.pkl'
dill.dump_session(filename)
dill.load_session(filename)

import tensorflow as tf
def f2_score(y_true, y_pred):
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 5 * precision * recall / (4 * precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)

Ytrain = to_categorical(Ytrain)
Yvaild = to_categorical(Yvaild)

model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape= (128,128,3)))

model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size = (3,3), activation= 'relu', padding= 'same'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size = (3,3), activation= 'relu', padding= 'same'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size = (3,3), activation= 'relu', padding= 'same'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, kernel_size = (3,3), activation= 'relu', padding= 'same'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation= 'relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation= 'relu'))
model.add(Dropout(0.25))
model.add(Dense(30, activation= 'softmax'))
optimizer = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss= 'categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', f2_score])
#early_stops = EarlyStopping(patience=3, monitor= 'val_acc')
checkpointer = ModelCheckpoint(filepath= 'weight.best.eda.hdf5', verbose= 1, save_best_only =True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

epochs = 30       # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86

datagen = ImageDataGenerator(featurewise_center=False,  samplewise_center=False, featurewise_std_normalization=False,  samplewise_std_normalization=False, zca_whitening=False, rotation_range=10, zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1,horizontal_flip=False, vertical_flip=False)
datagen.fit(Xtrain)


history = model.fit_generator(datagen.flow(Xtrain,Ytrain, batch_size=batch_size), epochs = epochs, validation_data = (Xvalid,Yvaild),verbose = 2, steps_per_epoch=Xtrain.shape[0] // batch_size  , callbacks=[learning_rate_reduction, checkpointer])
#model.fit(Xtrain, Ytrain, validation_data = (Xvalid, Yvaild), epochs =  10, batch_size = 100, callbacks= [checkpointer], verbose=1)

train_pred = model.predict(Xtrain).round()
f1_score(Ytrain, train_pred, average='samples')


valid_pred = model.predict(Xvalid).round()
f1_score(Yvaild, valid_pred, average = 'samples')

model.save('first_model.h5')



#predicting the test data
test_pred = model.predict(X_norm_test)

Predict_data = pd.DataFrame(np.concatenate((test.values, test_pred), axis = 1))
Predict_data.columns = [ 'image_id','antelope','bat','beaver','bobcat','buffalo','chihuahua','chimpanzee','collie','dalmatian','german+shepherd','grizzly+bear','hippopotamus','horse','killer+whale','mole','moose','mouse','otter','ox','persian+cat','raccoon','rat','rhinoceros','seal','siamese+cat','spider+monkey','squirrel','walrus','weasel','wolf']
Predict_data.head()
Predict_data.to_csv('predict_data.csv', index=False)
