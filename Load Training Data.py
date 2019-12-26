#!/usr/bin/env python
# coding: utf-8

# In[17]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

Datasets = 'C:\\Users\\Faizal\\Documents\\FYP\\Dataset Running Shoes\\Training'
CATEGORIES = ["Low","Normal","High"]

training_data = []

IMG_HEIGHT = 200
IMG_WIDTH = 125

for category in CATEGORIES:
    path = os.path.join(Datasets,category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
            training_data.append([new_array,class_num])
        except Exception as e:
            pass


# In[18]:


print(len(training_data))


# In[19]:


#random.shuffle(training_data)
#for sample in training_data[:]:
 #   print(sample[1])


# In[22]:


X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1,IMG_HEIGHT,IMG_WIDTH,1)
y = np.array(y)
pickle_out = open("train_image.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("train_label.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()


# In[23]:


pickle_in = open("train_image.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("train_label.pickle","rb")
y = pickle.load(pickle_in)


# In[ ]:





# In[ ]:




