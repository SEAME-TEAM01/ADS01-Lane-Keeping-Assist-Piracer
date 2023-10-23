# ------------------------------------------------------------------------------
# Third-party Library Import
import  matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
def plot_fit(history):
    # Learning curve visualize
    # - Loss Graph
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Evolution')
    # - Accuracy Graph
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy Evolution')
    plt.tight_layout()
    plt.show()