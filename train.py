# train.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

IMG_SIZE = (224,224)
batch = 32
train_dir = 'data/train'
val_dir = 'data/val'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_ds = train_datagen.flow_from_directory(train_dir,
                                         target_size=IMG_SIZE,
                                         class_mode='categorical',
                                         batch_size=batch)
val_ds = val_datagen.flow_from_directory(val_dir,
                                     target_size=IMG_SIZE,
                                     class_mode='categorical',
                                     batch_size=batch)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=20, validation_data=val_ds)
model.save('model/food_classifier.h5')
