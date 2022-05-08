# cnn-classification-covid19
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import metrics
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16  

import os
train_normal_dir_xray= os.path.join(r'C:\Users\Rahmani\Desktop\apprentissage 1\Noncovid\xray')
train_covide_dir_xray = os.path.join(r'C:\Users\Rahmani\Desktop\apprentissage 1\Covid\xray')
test_normal_dir_xray = os.path.join(r'C:\Users\Rahmani\Desktop\Test1\Noncovid\xray')
test_covide_dir_xray = os.path.join(r'C:\Users\Rahmani\Desktop\Test1\covid\xray')
train_normal_names = os.listdir(train_normal_dir_xray)
train_covide_names = os.listdir(train_covide_dir_xray)
test_normal_names = os.listdir(test_normal_dir_xray)
test_covide_names = os.listdir(test_covide_dir_xray)
print('total train normal chest xray: ', len(os.listdir(train_normal_dir_xray)))
print('total train covide xray:', len(os.listdir(train_covide_dir_xray)))
print('total test normal chest xray: ', len(os.listdir(test_normal_dir_xray)))
print('total test covide chest xray: ', len(os.listdir(test_covide_dir_xray)))


# define input image
input_shape = (299,299,3)

# Input layer
img_imput = Input(shape  = input_shape, name = 'img_input')

# Convo layers
x = Conv2D(32, (3,3) , padding = 'same' , activation='relu', name = 'layer_1' ,) (img_imput)
x = Conv2D(64, (3,3) , padding = 'same' , activation='relu', name = 'layer_2') (x)
x = MaxPool2D((2,2), strides=(2,2), name = 'layer_3') (x)
x = Dropout(0.25)(x)

x = Conv2D(64, (3,3) , padding = 'same' , activation='relu', name = 'layer_4') (x)
x = MaxPool2D((2,2), strides=(2,2), name = 'layer_5') (x)
x = Dropout(0.25)(x)

x = Conv2D(128, (3,3) , padding = 'same' , activation='relu', name = 'layer_6') (x)
x = MaxPool2D((2,2), strides=(2,2), name = 'layer_7') (x)
x = Dropout(0.25)(x)

x = Flatten(name = 'fc_1')(x)
x= Dense(64, name = 'lyaer_8')(x)
x = Dropout(0.5) (x)
x = Dense(2, activation='sigmoid', name='predictions')(x)

model = Model(inputs = img_imput, outputs =x , name='Binary_classification' )
model.summary()


# start Train/Test

batch_size = 64
hist = model.fit(traindata, 
                  steps_per_epoch = traindata.samples//batch_size,
                  validation_data = testdata,
                  validation_steps = testdata.samples//batch_size,
                epochs = 5
                
                 )
                 
                 plt.plot(hist.history['loss'], label = 'train')
plt.plot(hist.history['val_loss'], label = 'val')
plt.title('CNN_In_8_Steps :  Loss  &  Validation Loss')
plt.legend()
plt.show()

plt.plot(hist.history['accuracy'], label = 'train')
plt.plot(hist.history['val_accuracy'], label = 'val')
plt.title('CNN_In_8_Steps :  Accuracy  &  Validation Accuracy')
plt.legend()
plt.show()




# Confusion Matrix  & Pres  & Recall   & F1-Score

target_names = ['Abnormal', 'Normal']
label_names = [0,1]

Y_pred = model.predict_generator(testdata)
y_pred = np.argmax(Y_pred ,  axis = 1)

cm = confusion_matrix(testdata.classes, y_pred, labels = label_names)


print('Confusion Matrix')
print(confusion_matrix(testdata.classes, y_pred))

print('classification_Report')
print(classification_report(testdata.classes, y_pred, target_names=target_names))

disp = ConfusionMatrixDisplay(confusion_matrix= cm, display_labels=target_names)
disp = disp.plot(cmap=plt.cm.Blues, values_format = 'g')
plt.show()

