import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections
from tensorflow.keras import layers, models, Input

print("TensorFlow version:", tf.__version__)
print("Setup complete and ready to go!")

# ── Load & preprocess data ──────────────────────────────────────────────────
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

# ── Class names ─────────────────────────────────────────────────────────────
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ── Dataset exploration ─────────────────────────────────────────────────────
print("\n=== CIFAR-10 Dataset Summary ===")
print(f"Training images: {x_train.shape[0]}")
print(f"Test images:     {x_test.shape[0]}")
print(f"Image size:      {x_train.shape[1]}x{x_train.shape[2]} pixels")
print(f"Colour channels: {x_train.shape[3]} (RGB)")
print(f"Number of classes: {len(class_names)}")
print(f"Classes: {class_names}")

train_counts = collections.Counter(y_train.flatten())
print("\nImages per class (training):")
for i, name in enumerate(class_names):
    print(f"  {name}: {train_counts[i]}")

# ── Sample images ───────────────────────────────────────────────────────────
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i][0]])
plt.suptitle("Sample CIFAR-10 Images", fontsize=16)
plt.show()

# ── Build CNN model ─────────────────────────────────────────────────────────
model = models.Sequential([
    Input(shape=(32, 32, 3)),

    # First convolution block
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Second convolution block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Third convolution block
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Flatten and classify
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes
])

model.summary()
