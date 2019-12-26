#!/usr/bin/env python
# coding: utf-8

# In[9]:


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

Datasets = 'C:\\Users\\Faizal\\Documents\\FYP\\Dataset Running Shoes\\Testing'
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


# In[10]:


print(len(training_data))


# In[11]:


test_image = []
test_label = []

for features,label in training_data:
    test_image.append(features)
    test_label.append(label)

test_image = np.array(test_image).reshape(-1,IMG_HEIGHT,IMG_WIDTH,1)

pickle_out = open("test_image.pickle","wb")
pickle.dump(test_image,pickle_out)
pickle_out.close()

pickle_out = open("test_label.pickle","wb")
pickle.dump(test_label,pickle_out)
pickle_out.close()


# In[12]:


pickle_in = open("test_image.pickle","rb")
test_image = pickle.load(pickle_in)

pickle_in = open("test_label.pickle","rb")
test_label = pickle.load(pickle_in)


# In[ ]:





# In[ ]:




