#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

# Data preprocessing and generator setup
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(r"D:\extract\train", target_size=(64, 64), batch_size=32, class_mode='binary')

# Model setup
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(train_generator, epochs=10, batch_size=16)

# Function to load and preprocess an image for prediction
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

# Testing the model on a new image
new_image_path =r"D:\extract\test\cats\cat_56.jpg"

new_image = load_and_preprocess_image(new_image_path)
prediction = model.predict(new_image)

# Display the original image
img = plt.imread(new_image_path)
plt.imshow(img)
plt.axis('off')
plt.show()

# Display the prediction
if prediction[0][0] > 0.5:
    print("It's a Dog!")
else:
    print("It's a Cat!")


# In[6]:


# Test the model on a new image
new_image_path = r"D:\extract\test\dogs\dog_155.jpg"

  # use a raw string
new_image = load_and_preprocess_image(new_image_path)
prediction = model.predict(new_image)

# Display the original image
img = plt.imread(new_image_path)
plt.imshow(img)
plt.axis('off')
plt.show()

# Display the prediction
if prediction[0][0] > 0.5:
    print("It's a Dog!")
else:
    print("It's a Cat!")


# In[ ]:




