# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 22:06:51 2020

@author: Gaurav
"""
from tensorflow.keras.applications.vgg16 import VGG16
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Model


train_path='E:/All Data Set/Achived/Classification/fruits-360/Training'
test_path='E:/All Data Set/Achived/Classification/fruits-360/Test'

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

no_of_class=glob("E:/All Data Set/fruits-360/Training/*")

vgg=VGG16(input_shape=[224,224,3],weights='imagenet', include_top=False)

for layer in vgg.layers:
  layer.trainable = False

# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(no_of_class), activation='softmax')(x)


# prediction=Dense(units=len(no_of_class),activation='softmax')(vgg.output)

model=Model(inputs=vgg.input,outputs=prediction)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(train,validation_data=test,epochs=5,steps_per_epoch=len(train),validation_steps=len(test))

model.save('fruits-360.model')


















