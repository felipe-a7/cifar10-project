"""
train_baseline.py

This file does three things:
1. Loads and preprocesses the CIFAR-10 dataset.
2. Trains the baseline CNN model on the training data.
3. Evaluates the trained model on the test set and saves it
   to the outputs folder.

Why this file exists:
- It keeps the full baseline CNN workflow in one place.
- It combines model training, evaluation, and saving.
- It makes the baseline experiment easier to run and manage.

How to run:
    python scripts/train_baseline.py

What you will see:
- Training progress for each epoch
- Final test accuracy printed in the terminal
- A saved model file in:
    outputs/CNN.keras

How to inspect the saved model later:
    python

Then inside Python:
    from tensorflow.keras.models import load_model
    model = load_model("outputs/CNN.keras")
    model.summary()
"""


import os
import sys

# Set the project root so imports from the src folder work correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_loader import load_cifar10
from src.model_CNN import build_baseline_cnn


def train_baseline_model():
    """
    Load CIFAR-10, train the baseline CNN, evaluate it,
    and save the trained model.

    Returns:
        model: Trained CNN model
        history: Training history returned by model.fit()
    """
    print("Starting baseline CNN training...")

    # Load CIFAR-10 dataset
    x_train, y_train, x_test, y_test, class_names = load_cifar10()

    # Build the baseline CNN model
    model = build_baseline_cnn()

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    print("Baseline Test Accuracy:", test_acc)
    print("Baseline Test Loss:", test_loss)

    # Create outputs folder if it does not exist
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Save the trained model
    save_path = os.path.join(output_dir, "CNN.keras")
    model.save(save_path)

    print(f"Saved model to: {save_path}")

    return model, history


def main():
    """
    Main function:
    - loads the dataset
    - trains the baseline CNN
    - evaluates the model
    - saves the trained model
    """
    train_baseline_model()


# Run the script only when executed directly
if __name__ == "__main__":
    main()