#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import tensorflow as tf
import numpy as np
import csv
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CATEGORIES = ["Low","Normal","High"]

def prepare(filepath):
    IMG_HEIGHT = 200
    IMG_WIDTH = 125
    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
    new_array = np.array(new_array).reshape(-1,IMG_HEIGHT,IMG_WIDTH,1)
    new_array = new_array/255.0
    return new_array


model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare('C:\\Users\\Faizal\\Documents\\FYP\\Dataset Running Shoes\\Real-life-footprint\\IMG20191112212210.jpg')])
print(prediction)

for i in prediction :
    if np.argmax(i) == 0 :
        arch = "Low"
    elif np.argmax(i) == 1 :
        arch = "Normal"
    else :
        arch = "High"
    print(arch)


# In[17]:


brand = 'Adidas'
gender = 'Women'
colour = 'White'

def Running_Shoe(arch,brand,gender,colour) :
        a=arch
        b=brand
        c=gender
        d=colour
        if b == "Adidas":
            variable = pd.read_table("C:\\Users\\Faizal\\Documents\\FYP\\Dataset Running Shoes\\Adidas.csv",sep=",")
        elif b == "Asics":
            variable = pd.read_table("C:\\Users\\Faizal\\Documents\\FYP\\Dataset Running Shoes\\Asics.csv",sep=",")
        elif b == "Mizuno":
            variable = pd.read_table("C:\\Users\\Faizal\\Documents\\FYP\\Dataset Running Shoes\\Mizuno.csv",sep=",")
        elif b == "Nike":
            variable = pd.read_table("C:\\Users\\Faizal\\Documents\\FYP\\Dataset Running Shoes\\Nike.csv",sep=",")
        else :
            variable = pd.read_table("C:\\Users\\Faizal\\Documents\\FYP\\Dataset Running Shoes\\Puma.csv",sep=",")
        
        run_shoe = variable.loc[(variable['Arch'] == a) & (variable['Brand'] == b) & (variable['Gender'] == c) &
                  (variable['Colour'] == d)]
        a_list = run_shoe['Shoes']
        
        b_list = []
        
        for i in a_list:
            b_list.append(str(i))
        return a_list


# In[18]:


shoe_list = Running_Shoe(arch,brand,gender,colour)

Shoe_img_path = 'C:\\Users\\Faizal\\Documents\\FYP\\Dataset Running Shoes\\Shoe Images'
complete_path = os.path.join(Shoe_img_path,brand,arch,gender,colour)


# In[19]:


for i in shoe_list:
    img = mpimg.imread(os.path.join(Shoe_img_path,brand,arch,gender,colour,i+".jpg"))
    imgplot = plt.imshow(img)
    plt.title(i)
    plt.show()


# In[ ]:





# In[ ]:




