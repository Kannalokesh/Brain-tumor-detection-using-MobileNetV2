import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import warnings
warnings.filterwarnings('ignore')

# Parameters
image_size = (224, 224)
channels=3
batch_size = 32
epochs = 25
data_path = 'C:/Users/junjo/Documents/python/projects/brain_tumor_detection/dataset'

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data generators
train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Verify class indices
print("\nClass indices:", train_generator.class_indices)


# Model architecture

base_model = MobileNetV2(
    input_shape=image_size + (3,),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

# Freeze base model layers initially
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(2, activation='softmax')
])

# Compile with lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint('best_brain_model.keras', save_best_only=True)
]

# Training
my_model = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)


# Prediction function
def predict_tumor(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=image_size)
    x = tf.keras.utils.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    
    pred = model.predict(x)
    return 'No tumor found' if np.argmax(pred) == 0 else 'Tumor detected!'

# Example usage
test_image = 'C:/Users/junjo/Documents/python/projects/brain_tumor_detection/test_data/pred/pred9.jpg'
print(predict_tumor(test_image))
