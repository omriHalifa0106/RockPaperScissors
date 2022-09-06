import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import model_from_json

widht = 300
height = 300

PATH = "C:\\Users\\halif\\programming\\Rock Paper Scissors\\Code\\dataset\\Rock-Paper-Scissors\\"
train_dir = os.path.join(PATH+"train")
train = keras.preprocessing.image_dataset_from_directory(train_dir,image_size=(widht,height),validation_split=0.2,seed=42,subset="training")
validation = keras.preprocessing.image_dataset_from_directory(train_dir,image_size=(widht,height),validation_split=0.2,seed=42,subset="validation")
test_dir = os.path.join(PATH,"test")
test = keras.preprocessing.image_dataset_from_directory(test_dir,image_size=(widht,height))


plt.figure(figsize=(10,10))

for images,labels in train.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train.class_names[labels[i]])
        plt.axis(False)

scaler = keras.Sequential([keras.layers.experimental.preprocessing.Rescaling(1/255)])
model = keras.Sequential([ scaler,
keras.layers.Conv2D(32,kernel_size=3,strides=2,input_shape=(300, 300, 3),activation="relu"),
keras.layers.MaxPooling2D(2),
keras.layers.Conv2D(64,kernel_size=3,strides=2,activation="relu"),
keras.layers.MaxPooling2D(2),
keras.layers.SpatialDropout2D(0.2),
keras.layers.Conv2D(128,kernel_size=3,strides=1,activation="relu"),
keras.layers.MaxPooling2D(2),
keras.layers.SpatialDropout2D(0.2),
keras.layers.Flatten(),
keras.layers.Dense(50,activation="relu"),
keras.layers.Dropout(0.2),
keras.layers.Dense(3,activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

history = model.fit(train,validation_data=validation,epochs=70)
model.summary()

acc=model.evaluate (test, verbose=1)[1]*100
print ('Model Accuracy on Test Set: ', acc)
history_df = pd.DataFrame(history.history)
plt.plot(history_df.accuracy,label="Accuracy")
plt.plot(history_df.val_accuracy,label="Validation Accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid()

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelRPS.h5")
print("Saved model to disk")