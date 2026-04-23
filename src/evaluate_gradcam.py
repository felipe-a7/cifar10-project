import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Set project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_loader import load_cifar10


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_trained_model(model_path=None):
    """
    Load a trained Keras model from outputs/.
    """
    if model_path is None:
        model_path = os.path.join(project_root, "outputs", "CNN_improved.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            f"Train a model first with:\n"
            f"python src/train_improved.py"
        )

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Force the model to build/call once
    dummy_input = tf.zeros((1, 32, 32, 3), dtype=tf.float32)
    _ = model(dummy_input, training=False)

    return model


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model and return predictions.
    """
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss:     {test_loss:.4f}")

    y_prob = model.predict(x_test, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    return test_loss, test_acc, y_true, y_pred, y_prob


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot and save confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix - CIFAR-10")
    plt.tight_layout()

    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"Confusion matrix saved to: {save_path}")


def print_classification_report(y_true, y_pred, class_names):
    """
    Print and save classification report.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:\n")
    print(report)

    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "classification_report.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Classification report saved to: {save_path}")


def show_prediction_examples(x_test, y_true, y_pred, y_prob, class_names, correct=True, num_examples=5):
    """
    Show and save correct or incorrect prediction examples with confidence.
    """
    if correct:
        indices = np.where(y_true == y_pred)[0]
        title_text = "Correct Predictions"
        save_name = "correct_predictions.png"
    else:
        indices = np.where(y_true != y_pred)[0]
        title_text = "Incorrect Predictions"
        save_name = "incorrect_predictions.png"

    if len(indices) == 0:
        print(f"No {title_text.lower()} found.")
        return

    chosen = indices[:num_examples]

    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(chosen):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(x_test[idx])
        pred_class = class_names[y_pred[idx]]
        true_class = class_names[y_true[idx]]
        confidence = y_prob[idx][y_pred[idx]] * 100

        plt.title(
            f"Pred: {pred_class}\nTrue: {true_class}\n{confidence:.1f}%",
            fontsize=10
        )
        plt.axis("off")

    plt.suptitle(title_text, fontsize=18)
    plt.tight_layout()

    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"{title_text} figure saved to: {save_path}")


def find_last_conv_layer_index(model):
    """
    Find the index of the last Conv2D layer in the model.
    """
    for i in range(len(model.layers) - 1, -1, -1):
        if isinstance(model.layers[i], tf.keras.layers.Conv2D):
            return i
    raise ValueError("No Conv2D layer found in model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_index, pred_index=None):
    """
    Generate a Grad-CAM heatmap for one image.

    This version is robust for Sequential models:
    - first gets the output of the last conv layer
    - then manually passes it through the remaining layers
    - computes gradients of the target class with respect to conv activations
    """
    # Model up to the last conv layer
    conv_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.layers[last_conv_layer_index].output
    )

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array, training=False)
        tape.watch(conv_outputs)

        x = conv_outputs
        for layer in model.layers[last_conv_layer_index + 1:]:
            x = layer(x, training=False)

        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise ValueError("Gradients could not be computed for Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def display_gradcam(x_test, y_true, y_pred, model, class_names, num_examples=5):
    """
    Show and save Grad-CAM for a few test images.
    """
    last_conv_layer_index = find_last_conv_layer_index(model)
    last_conv_layer_name = model.layers[last_conv_layer_index].name
    print(f"Using last conv layer: {last_conv_layer_name}")

    chosen = np.arange(min(num_examples, len(x_test)))

    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(chosen):
        img = x_test[idx]
        img_array = np.expand_dims(img, axis=0).astype("float32")

        heatmap = make_gradcam_heatmap(
            img_array,
            model,
            last_conv_layer_index,
            pred_index=y_pred[idx]
        )

        plt.subplot(2, num_examples, i + 1)
        plt.imshow(img)
        plt.title(
            f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}",
            fontsize=9
        )
        plt.axis("off")

        plt.subplot(2, num_examples, num_examples + i + 1)
        plt.imshow(img)
        plt.imshow(heatmap, cmap="jet", alpha=0.4)
        plt.title("Grad-CAM", fontsize=9)
        plt.axis("off")

    plt.tight_layout()

    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "gradcam_examples.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"Grad-CAM examples saved to: {save_path}")


def print_confusion_analysis(y_true, y_pred, class_names, top_n=5):
    """
    Print the most confused class pairs.
    """
    cm = confusion_matrix(y_true, y_pred)
    confusion_pairs = []

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((cm[i, j], class_names[i], class_names[j]))

    confusion_pairs.sort(reverse=True, key=lambda x: x[0])

    print(f"\nTop {top_n} most common confusions:")
    for count, true_class, pred_class in confusion_pairs[:top_n]:
        print(f"  True {true_class:>10} → Pred {pred_class:<10} : {count}")


def main():
    _, _, x_test, y_test, class_names = load_cifar10()

    model = load_trained_model()

    _, _, y_true, y_pred, y_prob = evaluate_model(model, x_test, y_test)

    plot_confusion_matrix(y_true, y_pred, class_names)
    print_classification_report(y_true, y_pred, class_names)
    print_confusion_analysis(y_true, y_pred, class_names, top_n=5)

    show_prediction_examples(
        x_test, y_true, y_pred, y_prob, class_names,
        correct=True, num_examples=5
    )

    show_prediction_examples(
        x_test, y_true, y_pred, y_prob, class_names,
        correct=False, num_examples=5
    )

    display_gradcam(
        x_test, y_true, y_pred, model, class_names, num_examples=5
    )


if __name__ == "__main__":
    main()