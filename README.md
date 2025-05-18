# 🧠 Image Classification using TensorFlow Hub

This project uses a **pre-trained MobileNetV2 model** from TensorFlow Hub to classify images.

## 🚀 How to Run

### Step 1: Install dependencies

```bash
pip install tensorflow tensorflow-hub matplotlib pillow
```

### Step 2: Add an image

Put an image inside the `images/` folder and rename it to `sample.jpg` (or change the path in the script).

### Step 3: Run the classifier

```bash
python classify.py
```

It will display the image and print the top prediction.

## 🖼️ Sample Output

![Sample](images/sample.jpg)

## 📦 Model Used

- [MobileNetV2 (TF Hub)](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4)
