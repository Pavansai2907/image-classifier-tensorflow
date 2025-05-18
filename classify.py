import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_image(image_path, image_size=(224, 224)):
    img = Image.open(image_path).convert("RGB").resize(image_size)
    img = np.array(img) / 255.0  # Normalize
    img = img[np.newaxis, ...]   # Add batch dimension
    return img

def main():
    # Load pre-trained model from TensorFlow Hub
    model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url, input_shape=(224, 224, 3))
    ])

    # Load ImageNet labels
    labels_path = tf.keras.utils.get_file(
        "ImageNetLabels.txt",
        "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    )
    with open(labels_path, "r") as f:
        labels = f.read().splitlines()

    # Path to the image
    image_path = "images/sample.jpg"  # Replace with your own image
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        return

    # Preprocess and predict
    image = load_image(image_path)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    predicted_label = labels[predicted_class]

    # Show result
    plt.imshow(Image.open(image_path))
    plt.axis('off')
    plt.title(f"Prediction: {predicted_label}")
    plt.show()

if __name__ == "__main__":
    main()
