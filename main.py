import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Path to your dataset
dataset_path = 'C:/Users/Admin/Downloads/devanagiri archive/DevanagariHandwrittenCharacterDataset/Train'

# Ensure the dataset path is correct
if not os.path.exists(dataset_path):
    raise ValueError(f"Dataset path {dataset_path} does not exist.")

try:
    train_dataset = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(32,32),
        batch_size=32,
        color_mode='grayscale'  
    )
    print("Training dataset loaded successfully.")
except Exception as e:
    print(f"Error loading training dataset: {e}")

try:
    validation_dataset = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(32,32),
        batch_size=32,
        color_mode='grayscale'  
    )
    print("Validation dataset loaded successfully.")
except Exception as e:
    print(f"Error loading validation dataset: {e}")

# Normalize the dataset
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(32,32, 1)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(46, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, validation_data=validation_dataset, epochs=3)

model.save('AI_Epigraphy.keras')

Load the model
model = tf.keras.models.load_model('AI_Epigraphy.keras')

def predict_img(image_path, model, class_names):
    try:
        print(f"Processing image: {image_path}")
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Error reading image {image_path}")
            return

        res_img = cv2.resize(img, (32,32))

        Norm_img = res_img / 255.0

        input_img = np.expand_dims(Norm_img, axis=0)
        input_img = np.expand_dims(input_img, axis=-1)

        print(f"Image shape for prediction: {input_img.shape}")
        prediction = model.predict(input_img)
        predicted_class = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class]

        print(f"This character is probably a {predicted_class_name}")
        
        _, thresholded_image = cv2.threshold(res_img, 127, 255, cv2.THRESH_BINARY)

        plt.figure(figsize=(5,5))
        plt.title(f"Predicted: {predicted_class_name}")
        plt.imshow(thresholded_image, cmap='gray')
        plt.axis('off')
        plt.show()
        tf.keras.backend.clear_session()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

class_names = sorted(os.listdir(dataset_path))

test_images = [
    'C:/Users/Admin/Downloads/devanagiri archive/DevanagariHandwrittenCharacterDataset/Test/character_1_ka/1339.png',
    #'C:/Users/Admin/Downloads/devanagiri archive/DevanagariHandwrittenCharacterDataset/Test/character_7_chha/4214.png',
    'C:/Users/Admin/Downloads/devanagiri archive/DevanagariHandwrittenCharacterDataset/Test/character_7_chha/4236.png',
    'C:/Users/Admin/Downloads/devanagiri archive/DevanagariHandwrittenCharacterDataset/Test/character_13_daa/580.png',
    'C:/Users/Admin/Downloads/devanagiri archive/DevanagariHandwrittenCharacterDataset/Test/digit_5/5414.png',
    'C:/Users/Admin/OneDrive/Desktop/EpigraphyAI/output_boxes/box_21.jpg',

]

for image_path in test_images:
    predict_img(image_path, model,class_names)

