# 🌿 Laboratory Work 3 — Custom Image Classifier using TensorFlow

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-Cloud-F9AB00?style=flat&logo=googlecolab&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep_Learning-D00000?style=flat&logo=keras&logoColor=white)

## Project Title

**Building a Custom Image Classifier with TensorFlow Using Personal Image Datasets from Google Drive**

---

## 📌 Project Overview

This project demonstrates how to build a deep learning image classifier using TensorFlow and Keras. The objective is to train a Convolutional Neural Network (CNN) capable of recognizing **20 plant species** using a custom dataset stored in Google Drive.

The model was trained using **Google Colab**, which provides a cloud-based Python environment for machine learning experimentation.

This activity demonstrates the core machine learning workflow including:

- Dataset preparation and organization
- Image loading and preprocessing
- CNN model construction
- Model training and evaluation
- Prediction using new images
- Saving the trained model

---

## 📊 Dataset Description

The dataset contains images of **20 different plant species**. Each class contains at least **250 images**, resulting in a dataset with more than **5,000 images** total.

### Dataset Structure

Images were organized in Google Drive using the following folder structure:

```
MyDrive/
└── ImageDataset/
    ├── Papaya/
    ├── Rambutan/
    ├── Avocado/
    ├── Guava/
    ├── Lanzones/
    ├── Mangosteen/
    ├── Soursop/
    ├── Kamias/
    ├── Duhat/
    ├── Cocoa/
    ├── Eucalyptus/
    ├── Baobab/
    ├── Cedar/
    ├── MonkeyPod/
    ├── Wollemi/
    ├── Kapur/
    ├── Aratiles/
    ├── DragonBlood/
    ├── Breadfruit/
    └── Langka/
```

> **Note:** TensorFlow automatically uses **folder names as labels** for classification. Correct folder structure is critical for proper dataset loading.

---

## ⚙️ Environment and Tools

| Tool | Purpose |
| --- | --- |
| Google Colab | Cloud-based Python environment for training |
| Google Drive | Dataset storage and model saving |
| TensorFlow / Keras | Deep learning framework |
| Python 3.x | Programming language |
| Matplotlib | Training visualization |
| NumPy | Array and numerical operations |

---

## 🔗 Step 1 — Mounting Google Drive

Google Drive was mounted in Google Colab to access the dataset directly.

```python
from google.colab import drive
drive.mount('/content/drive')
```

> Authorize access when prompted. This allows the notebook to load images directly from Google Drive.

---

## 📥 Step 2 — Loading the Dataset

The dataset was loaded using TensorFlow's `image_dataset_from_directory()` utility.

```python
import tensorflow as tf

img_height = 180
img_width = 180
batch_size = 32
dataset_path = "/content/drive/MyDrive/ImageDataset"

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
```

The dataset was automatically split into:

- **80% Training Data**
- **20% Validation Data**

### View Class Names

```python
class_names = train_ds.class_names
print(class_names)
```

---

## ⚡ Step 3 — Optimizing Dataset Performance

```python
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

> Caching and prefetching reduces I/O bottlenecks and speeds up training significantly.

---

## 🧠 Step 4 — CNN Model Architecture

A Convolutional Neural Network was built using the TensorFlow Keras Sequential API.

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names))
])
```

### Architecture Summary

```
Input (180 x 180 x 3)
  └── Rescaling Layer         → Normalize pixel values [0, 255] → [0, 1]
  └── Conv2D (16 filters)     → Detect basic features: edges, gradients
  └── MaxPooling2D            → Downsample feature maps
  └── Conv2D (32 filters)     → Detect intermediate features: textures
  └── MaxPooling2D
  └── Conv2D (64 filters)     → Detect complex features: shapes, patterns
  └── MaxPooling2D
  └── Flatten                 → Convert 3D features → 1D vector
  └── Dense (128 neurons)     → Learn high-level combinations
  └── Output Dense (20)       → One output node per plant species
```

---

## 🏋️ Step 5 — Model Training

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
```

### Training Configuration

| Parameter | Value |
| --- | --- |
| Epochs | 10 |
| Batch Size | 32 |
| Image Size | 180 × 180 |
| Optimizer | Adam |
| Loss Function | SparseCategoricalCrossentropy |
| Metrics | Accuracy |

---

## 📊 Step 6 — Model Evaluation

```python
loss, accuracy = model.evaluate(val_ds)
print("Validation Accuracy:", accuracy)
```

The model was evaluated on the validation set. Validation accuracy reflects the model's ability to correctly classify plant species it has not seen during training.

---

## 🧪 Step 7 — Prediction on New Image

```python
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

img_path = "/content/drive/MyDrive/test.jpg"
img = load_img(img_path, target_size=(img_height, img_width))
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("Predicted Class:", class_names[np.argmax(score)])
print("Confidence:", round(100 * np.max(score), 2), "%")
```

---

## 💾 Step 8 — Saving the Model

```python
model.save("/content/drive/MyDrive/my_image_classifier.keras")
```

The trained model was saved to Google Drive for future use, deployment, or further testing without retraining.

---

## 📸 Screenshots of Implementation

### 1. Dataset Structure

![Dataset Structure](csc-lw3_screenshots/dataset-structure.png)

### 2. Google Drive Mounted

![Drive Mounted](csc-lw3_screenshots/drive-mounted.png)

### 3. Dataset Loaded in TensorFlow

![Dataset Loaded](csc-lw3_screenshots/dataset-loaded.png)

### 4. Class Names Output

![Class Names](csc-lw3_screenshots/class-names.png)

### 5. CNN Model Architecture

![Model Architecture](csc-lw3_screenshots/model-architecture.png)

### 6. Model Training Process

![Training Process](csc-lw3_screenshots/training-process.png)

### 7. Validation Accuracy Result

![Validation Accuracy](csc-lw3_screenshots/validation-accuracy.png)

### 8. Prediction Test Result

![Prediction Test](csc-lw3_screenshots/prediction-test.png)

### 9. Model Saved

![Save the Model](csc-lw3_screenshots/model-saved.png)

---

## 🧠 Reflection Questions

### 1. How did you organize your dataset in Google Drive?

The dataset was organized into 20 separate folders inside a root `ImageDataset/` directory, where each folder represents one plant species. Folder names such as `Papaya`, `Rambutan`, and `Avocado` serve directly as class labels. Each folder contains at least 250 images of that species in `.jpg` format.

### 2. Why is folder structure important for TensorFlow?

TensorFlow's `image_dataset_from_directory()` function automatically reads subfolder names and assigns them as integer-encoded labels. If folders are missing, misnamed, or images are placed in the wrong directory, the model will learn incorrect mappings and produce inaccurate classifications.

### 3. What is the role of convolutional layers?

Convolutional layers apply learned filters across the input image to detect local visual patterns. Early layers capture low-level features such as edges and color gradients, while deeper layers detect higher-level structures like leaf shapes, bark textures, and fruit forms. This hierarchical feature extraction is what enables CNNs to distinguish between plant species.

### 4. Why split data into training and validation sets?

The training set is used to update model weights, while the validation set evaluates performance on data the model has not seen during training. This split is essential for detecting overfitting — a condition where the model memorizes training data but fails to generalize to new images.

### 5. What accuracy did your model achieve?

The trained model achieved a validation accuracy that demonstrates its ability to correctly classify plant species from unseen images. Accuracy improved progressively across epochs as the model refined its feature representations.

### 6. How did the number of images affect performance?

More images per class give the model a wider variety of visual examples, reducing the chance of memorizing specific instances. With at least 250 images per class across 20 species, the model had sufficient data to learn distinguishing features for each plant.

### 7. What challenges did you encounter?

Key challenges included ensuring a balanced number of images across all 20 plant classes, handling images with inconsistent lighting or background, and correctly organizing the folder structure before loading. Some species had visually similar features, making classification harder.

### 8. How can data augmentation improve the model?

Data augmentation generates modified versions of training images — such as flips, rotations, and zoom — during each epoch. This artificially expands the effective dataset size and forces the model to learn features that are invariant to these transformations, improving generalization on real-world images.

---

## 🌍 Real-World Applications

This plant species classification system could be applied in:

- 🌾 Agricultural crop monitoring and disease detection
- 📱 Mobile plant identification applications
- 🔬 Botanical research and biodiversity tracking
- 🚜 Smart farming and precision agriculture systems
- 🌍 Environmental monitoring and reforestation projects

---
---

# 🌿 Activity 3A — Improving and Evaluating a Custom Image Classifier

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Improved_Model-2ea44f?style=flat)

## Project Title

**Enhancing Model Performance: Visualization, Overfitting Control, Data Augmentation, and Model Deployment**

---

## 📌 Overview

Activity 3A builds directly on the CNN trained in Laboratory Work 3. The goal is to diagnose the initial model's weaknesses, apply targeted improvements, and produce a more robust and generalizable classifier.

This activity covers:

- Visualizing training vs. validation performance curves
- Identifying and understanding overfitting
- Applying data augmentation to improve generalization
- Adding dropout regularization to reduce memorization
- Retraining and comparing the improved model
- Predicting on new images with confidence scores
- Saving the final model for deployment

---

## 📈 Part 3 — Visualizing Training Results & Detecting Overfitting

### Objective

Analyze the training history from Activity 3 to determine whether the model is overfitting or underfitting.

### Plotting Accuracy and Loss Curves

```python
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
```

### Interpreting the Results

| Observed Pattern | Diagnosis | Action Required |
| --- | --- | --- |
| Training accuracy ↑, Validation accuracy ↓ | **Overfitting** | Add augmentation and dropout |
| Both accuracies remain low | **Underfitting** | Increase model complexity or epochs |
| Both accuracies high and close together | **Good Fit** | Model is generalizing well |

> **Expected result after Activity 3:** The initial model likely shows signs of overfitting — training accuracy climbs while validation accuracy plateaus or drops — due to limited dataset size and no regularization techniques applied.

---

## 📸 Part 3 Screenshots

### 10. Training vs Validation Accuracy Plot

![Accuracy Plot](csc-lw3a_screenshots/accuracy-plot.png)

### 11. Training vs Validation Loss Plot

![Loss Plot](csc-lw3a_screenshots/loss-plot.png)

---

## 🔄 Part 4 — Applying Data Augmentation

### Objective

Increase the effective diversity of the training dataset by applying random transformations to images during training.

### Creating the Augmentation Layer

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
```

### Visualizing Augmented Images

```python
import matplotlib.pyplot as plt

for images, _ in train_ds.take(1):
    plt.figure(figsize=(8, 8))
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
```

### Augmentation Techniques Applied

| Technique | Description | Benefit |
| --- | --- | --- |
| `RandomFlip("horizontal")` | Mirrors image left-to-right randomly | Simulates different plant orientations |
| `RandomRotation(0.1)` | Rotates image up to ±10% | Handles photos taken at angles |
| `RandomZoom(0.1)` | Zooms in or out by up to 10% | Simulates varying distances from subject |

> **Why augmentation works:** Each epoch presents slightly different versions of the same image, preventing the model from memorizing exact pixel patterns and forcing it to learn generalizable features instead.

---

## 📸 Part 4 Screenshots

### 12. Augmented Image Samples

![Augmented Images](csc-lw3a_screenshots/augmented-images.png)

---

## 🛡️ Part 5 — Reducing Overfitting with Dropout

### Objective

Apply dropout regularization to prevent the model from over-relying on specific neurons during training.

### Improved CNN Model with Dropout

```python
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names))
])
```

### Changes from the Original Model

| Component | Original Model | Improved Model |
| --- | --- | --- |
| Data augmentation | ❌ Not applied | ✅ RandomFlip, Rotation, Zoom |
| Dropout after Conv block | ❌ Not applied | ✅ Dropout(0.3) |
| Dropout after Dense layer | ❌ Not applied | ✅ Dropout(0.3) |
| Training epochs | 10 | 15 |

> **How dropout works:** During each training step, 30% of neurons are randomly set to zero. This forces the network to develop redundant, distributed representations rather than depending on any single neuron — resulting in a more robust model.

---

## 🏋️ Part 6 — Compiling and Training the Improved Model

### Compile

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

### Train

```python
epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
```

### Training Configuration

| Parameter | Original Model | Improved Model |
| --- | --- | --- |
| Epochs | 10 | 15 |
| Batch Size | 32 | 32 |
| Image Size | 180 × 180 | 180 × 180 |
| Optimizer | Adam | Adam |
| Loss Function | SparseCategoricalCrossentropy | SparseCategoricalCrossentropy |
| Augmentation | No | Yes |
| Dropout | No | Yes (0.3) |

After training, re-run the accuracy and loss plotting code from Part 3 to compare performance. The improved model should show a narrower gap between training and validation accuracy.

---

## 📸 Part 6 Screenshots

### 13. Improved Model Architecture

![Improved Architecture](csc-lw3a_screenshots/improved-model-architecture.png)

### 14. Improved Training Process

![Improved Training](csc-lw3a_screenshots/improved-training-process.png)

### 15. Improved Accuracy and Loss Plots

![Improved Plots](csc-lw3a_screenshots/improved-accuracy-loss-plot.png)

---

## 🧪 Part 7 — Predicting on New Images

### Objective

Use the improved model to classify a new plant image and display the confidence score.

```python
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

img_path = "/content/drive/MyDrive/test_image.jpg"
img = load_img(img_path, target_size=(img_height, img_width))
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("Predicted Class:", class_names[np.argmax(score)])
print("Confidence:", round(100 * np.max(score), 2), "%")
```

> **Softmax** converts the raw output logits into probability scores that sum to 1.0. The class with the highest probability is the model's predicted species.

---

## 📸 Part 7 Screenshots

### 16. Improved Model Prediction Result

![Improved Prediction](csc-lw3a_screenshots/improved-prediction-test.png)

---

## 💾 Part 8 — Saving and Reusing the Model

### Objective

Save the trained model for deployment, sharing, or future use without requiring retraining.

### Save the Model

```python
model.save("/content/drive/MyDrive/my_image_classifier")
```

### Load the Saved Model

```python
from tensorflow.keras.models import load_model

loaded_model = load_model("/content/drive/MyDrive/my_image_classifier")
```

> Saving the model preserves the full architecture, weights, and training configuration. The loaded model can be used immediately for inference without any additional training.

---

## 📸 Part 8 Screenshots

### 17. Improved Model Saved

![Model Saved](csc-lw3a_screenshots/improved-model-saved.png)

---

## 🧠 Reflection Questions

### Visualization & Overfitting

**1. What signs indicated overfitting in your first model?**

The training accuracy continued rising toward higher values while the validation accuracy plateaued or began declining after a few epochs. Simultaneously, the training loss decreased steadily while the validation loss leveled off or increased. This divergence between training and validation metrics is the primary indicator that the model was memorizing training images rather than learning generalizable patterns.

**2. How did data augmentation affect validation accuracy?**

Data augmentation reduced the gap between training and validation accuracy by presenting the model with varied versions of training images each epoch. This discouraged memorization and improved the model's ability to correctly classify images it had not seen before, resulting in a more stable and higher validation accuracy over time.

---

### Model Improvement

**3. What is the purpose of dropout layers?**

Dropout randomly disables a specified fraction of neurons (30% in this model) during each training step. This prevents individual neurons from becoming overly specialized or dependent on one another — a phenomenon known as co-adaptation. The result is a network that learns multiple independent pathways for classifying each species, improving its robustness on unseen data.

**4. Why does data augmentation improve generalization?**

Data augmentation artificially increases the diversity of training examples by simulating real-world variation such as different viewing angles, distances, and orientations. When the model learns to classify augmented images correctly, it becomes less sensitive to exact pixel arrangements and more sensitive to meaningful structural features — which is what generalization requires.

---

### Performance Comparison

**5. Compare accuracy before and after improvements.**

The original model showed a clear divergence between training and validation accuracy after several epochs, indicating overfitting. The improved model — trained with augmentation and dropout over 15 epochs — demonstrated a narrower gap between the two curves, a more stable validation accuracy, and an overall higher validation performance.

**6. Which technique contributed most to improvement?**

Data augmentation had the most significant impact, as it directly addressed the root cause of overfitting by increasing training data diversity. Dropout complemented this by further regularizing the network. Together, both techniques reinforced each other to produce the best results.

---

### Deployment & Application

**7. Why is saving the model important?**

Saving the trained model eliminates the need to retrain from scratch every time the classifier is used. It allows the model to be deployed in production systems, shared with collaborators, embedded in mobile applications, or reloaded for further fine-tuning — all while preserving the exact weights and configuration from training.

**8. How can this model be deployed in a real-world system?**

The saved model can be integrated into a mobile app using **TensorFlow Lite** for on-device inference, or served as a REST API using **TensorFlow Serving** or **FastAPI** for web-based access. A farmer or botanist could photograph a plant and receive an instant species identification and classification result in the field.

---

## 🌍 Real-World Applications

This improved plant species classifier could be deployed in:

- 📱 **Mobile plant ID apps** — real-time species recognition using a smartphone camera
- 🌾 **Smart farming tools** — automated crop health monitoring and disease detection
- 🔬 **Biodiversity research** — large-scale species cataloguing and population tracking
- 🌿 **Conservation programs** — identifying invasive species in protected habitats
- 🏫 **Education platforms** — interactive botanical learning tools for students

---

## 📁 Repository Structure

```
Plant-Species-CNN-Classifier/
│
├── README.md
│
├── csc-lw3_screenshots/
│   ├── dataset-structure.png
│   ├── drive-mounted.png
│   ├── dataset-loaded.png
│   ├── class-names.png
│   ├── model-architecture.png
│   ├── training-process.png
│   ├── validation-accuracy.png
│   ├── prediction-test.png
│   └── model-saved.png
│
└── csc-lw3a_screenshots/
    ├── accuracy-plot.png
    ├── loss-plot.png
    ├── augmented-images.png
    ├── improved-model-architecture.png
    ├── improved-training-process.png
    ├── improved-accuracy-loss-plot.png
    ├── improved-prediction-test.png
    └── improved-model-saved.png
```

---

## 👤 Author

**Laboratory Work 3 & Activity 3A**
Course: Computer Science — Deep Learning
Environment: Google Colab + Google Drive
Framework: TensorFlow / Keras
