#!/usr/bin/env python
# coding: utf-8

# In[37]:


import tensorflow as tf
from flask import Flask, render_template
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

X = pickle.load(open("train_image.pickle","rb"))
y = pickle.load(open("train_label.pickle","rb"))

IMG_HEIGHT = 200
IMG_WIDTH = 125
 
batch_size = 10
epochs = 5

X = X/255.0
#X = X.astype(np.float32)
y = np.array(y)
y = y.astype(np.float32)


# In[38]:



'''model = Sequential()
model.add(Conv2D(64,(3,3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(3))
model.add(Activation('softmax'))'''

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))


# In[39]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[40]:


model.fit(X,y,batch_size=5,epochs=15,validation_split=0.2) 


# In[41]:


test_image = pickle.load(open("test_image.pickle","rb"))
test_label = pickle.load(open("test_label.pickle","rb")) 

test_image = (test_image/255.0)
#test_image = test_image.astype(np.float32)
test_label = np.array(test_label)
test_label = test_label.astype(np.float32)


# In[42]:


predictions = model.predict(test_image)


# In[43]:


predictions[12]
#0 - Low
#1 - Normal
#2 - High


# In[44]:


test_label[7]


# In[45]:


np.argmax(predictions[0])


# In[46]:


labels = []
for i in predictions :
    if np.argmax(i) == 0 :
        print("Low")
        labels.append(0)
    elif np.argmax(i) == 1 :
        print("Normal")
        labels.append(1)
    else :
        print("High")
        labels.append(2)


# In[47]:


print(labels)


# In[48]:


print(test_label)


# In[49]:


correct = 0

for i in range (len(test_label)) :
    b = 0    
    if labels[i] == test_label[i]:
        correct = correct + 1
        
b=b+1


# In[50]:


Recognition = 0.00

Recognition = (correct/len(test_label)) * 100
print("Accuracy : ",Recognition,"%")


# In[52]:


model.save('64x3-CNN.model')


# In[ ]:




