#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow  # installing tenserflow


# In[5]:


import zipfile
import os

# Specifying the path to cat and dog dataset
zip_file_path = r"C:\Users\ANUP\Downloads\archive (3).zip"

# Specifing the directory where to extract the contents
extracted_dir = r"D:\extract"

# Creating the directory if it doesn't exist
os.makedirs(extracted_dir, exist_ok=True)

# Extracting the contents of the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)

# Now there are two separate folders for training and testing data in 'extracted_dir'

train_dir = os.path.join(extracted_dir, 'train')
test_dir = os.path.join(extracted_dir, 'test')

# we can access the cat and dog images within these folders for further processing


# In[7]:


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[10]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    r"D:\extract\train",  # Corrected path with raw string
    target_size=(64, 64),  # adjust size as needed
    batch_size=32,
    class_mode='binary'  # 'categorical' for more than two classes
)


# In[14]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # 'sigmoid' for binary classification


# In[15]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[18]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% of the data will be used for validation
)

train_generator = train_datagen.flow_from_directory(
    r"D:\extract\train",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='training'  # This is the training set
)

val_generator = train_datagen.flow_from_directory(
    r"D:\extract\train",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # This is the validation set
)

model.fit(train_generator, epochs=10, validation_data=val_generator)



# In[20]:


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
     r"D:\extract\test",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy*100:.2f}%")


# In[22]:


from sklearn.metrics import classification_report, confusion_matrix

# Get predictions
predictions = model.predict(test_generator)
y_pred = (predictions > 0.5).astype(int)

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(test_generator.classes, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(test_generator.classes, y_pred, target_names=['cat', 'dog']))


# In[ ]:




