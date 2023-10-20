import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


WIDTH=640
HEIGHT=480

model_str = "model4.h5"
loaded_model = load_model(model_str)
data_df = pd.read_csv("dataset/record.csv")
image_path = "dataset/frames/"

def preprocess_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (WIDTH, HEIGHT)) / 255.0
    image = np.expand_dims(image, axis=-1)
    return image

def test_random_image(model, df, image_path):
    # Randomly select a row from the dataframe
    random_row = df.sample().iloc[0]
    
    # Get the image and label
    image_filename = f"frame_{random_row['index']}_{random_row['steering']}.jpg"
    full_image_path = os.path.join(image_path, image_filename)
    
    test_image = preprocess_image(full_image_path)
    test_image_batch = np.expand_dims(test_image, axis=0)
    
    # Predict
    predictions = model.predict(test_image_batch)
    predicted_label = np.argmax(predictions, axis=1)[0]
    
    # Get the actual label
    if random_row['direction(front-0/left-1/right-2)'] == 0:
        actual_label = "Front"
    elif random_row['direction(front-0/left-1/right-2)'] == 1:
        actual_label = "Left"
    elif random_row['direction(front-0/left-1/right-2)'] == 2:
        actual_label = "Right"
    
    # Get the predicted label
    if predicted_label == 0:
        predicted_text = "Front"
    elif predicted_label == 1:
        predicted_text = "Left"
    elif predicted_label == 2:
        predicted_text = "Right"

    return actual_label, predicted_text

# Test
x = 0
print(f"idx  actual  predicted  result")
results = []
actual_labels = []
predicted_labels = []

for x in range(0, 200):
    actual, predicted = test_random_image(loaded_model, data_df, image_path)
    result = actual == predicted
    results.append(result)
    actual_labels.append(actual)
    predicted_labels.append(predicted)
    print(f"{x:3d}  {actual:5s}  {predicted:5s}  {result}")

print(f"final result : {np.count_nonzero(results)}/{(len(results))}")

cm = confusion_matrix(actual_labels, predicted_labels, labels=["Front", "Left", "Right"])

# Confusion Matrix 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Front", "Left", "Right"], yticklabels=["Front", "Left", "Right"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Precision, Recall, F1-Score
report = classification_report(actual_labels, predicted_labels, target_names=["Front", "Left", "Right"])
print(report)
