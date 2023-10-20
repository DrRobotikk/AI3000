import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Use your model instead of the pre-defined model
model = MobileNetV2(weights='imagenet')

# Function to load and preprocess an image from a file path
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Inference on new images
new_image_path = 'C:\\Users\\ws\\Desktop\\Test.jpg'  # Replace with the path to your new image

# Load and preprocess the new image
new_image = load_and_preprocess_image(new_image_path)

# Get predictions for the new image
predictions = model.predict(new_image)
decoded_predictions = decode_predictions(predictions, top=5)[0]

# Display the top 5 predicted classes
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print("Prediction {}: {} ({:.2f}%)".format(i + 1, label, score * 100))

# Display the image
img = image.load_img(new_image_path)
plt.imshow(img)
plt.axis('off')
plt.show()


