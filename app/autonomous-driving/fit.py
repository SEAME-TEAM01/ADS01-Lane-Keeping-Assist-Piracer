import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("- data load")
# 데이터 로드
data = pd.read_csv('dataset/record.csv')
img_height, img_width = 480, 640  # 이미지 크기 (적절한 크기로 수정 가능)

# 이미지와 라벨을 로드하는 함수
def load_data(data, img_height, img_width):
    images = []
    labels = []
    for idx, row in data.iterrows():
        img_path = f"dataset/frames/frame_{row['index']}_{row['steering']}.jpg"
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        # 필터링: 흑백으로 변환 후 블러 처리
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # 이미지 리사이즈
        img = cv2.resize(img, (img_width, img_height))
        
        # 정규화
        img = img / 255.0
        images.append(img)
        labels.append(row['direction(front-0/left-1/right-2)'])
        
    labels = to_categorical(labels, num_classes=3)  # 원-핫 인코딩
        
    return np.array(images), np.array(labels)

print("- image load & reshape")
X, y = load_data(data, img_height, img_width)
X = X.reshape(X.shape[0], img_height, img_width, 1)
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print("- label analysis")
# 레이블 분포 분석
unique, counts = np.unique(y_train, return_counts=True)
label_distribution = dict(zip(unique, counts))
print("Label distribution in training data:", label_distribution)

print("- CNN model build")
# CNN 모델 구성
model = Sequential([
    Conv2D(24, (5,5), activation='relu', input_shape=(img_height, img_width, 1)),
    MaxPooling2D((2,2)),
    Conv2D(36, (5,5), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(48, (5,5), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3개의 클래스: 앞/왼쪽/오른쪽
])

print("- model compile & train")
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32, verbose=1)
model.save('model4.h5')

print("- plot")
# 학습 과정 시각화
# 손실 그래프
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Evolution')

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Evolution')

plt.tight_layout()
plt.show()
