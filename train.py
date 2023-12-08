# need to change it to Annotation YOLO only.
# need to change it to Annotation YOLO only.
# need to change it to Annotation YOLO only.
# need to change it to Annotation YOLO only.
# need to change it to Annotation YOLO only.
# need to change it to Annotation YOLO only.
# need to change it to Annotation YOLO only.
# need to change it to Annotation YOLO only.
# need to change it to Annotation YOLO only.
# need to change it to Annotation YOLO only.

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from yolov4.tf import YOLOv4, YOLODataset
from tensorflow.keras.callbacks import Callback

count = 1
count2 = 1

# CSV 파일에서 데이터 읽어오기
csv_path = "/Users/minsik/development/Blind_Helper/Annotations/training_file.csv"
df = pd.read_csv(csv_path)

# Label이 0 또는 1인 데이터만 선택
df = df[df['mode'].isin([0, 1])]

# 처음부터 1000개의 데이터만 사용
df = df.head(200)

# 이미지 경로, 박스 좌표 및 라벨 추출
image_paths = df['file'].values
labels = df['mode'].values
boxes = df[['x1', 'y1', 'x2', 'y2']].values

# 이미지 리사이즈 및 박스 좌표 조정 함수
def resize_and_adjust_boxes(image_path, box, label, resize_factor):
    global count  # 전역 변수로 선언
    img = cv2.imread("/Users/minsik/development/Blind_Helper/Img/" + image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 이미지 크기를 4분의 1로 줄임
    new_size = (img.shape[1] // resize_factor, img.shape[0] // resize_factor)
    img = cv2.resize(img, new_size)

    # 박스 좌표 조정
    box = [coord // resize_factor for coord in box]

    # 이미지를 0과 1 사이의 값으로 정규화
    img = img / 255.0
    print("Resized " + str(count) + " Imgs")
    count += 1  # global 키워드 사용하여 값을 증가시킴
    return img, box, label

# 데이터셋 생성
images = []
adjusted_boxes = []
for path, box, label in zip(image_paths, boxes, labels):
    img, adjusted_box, label = resize_and_adjust_boxes(path, box, label, 4)
    images.append(img)
    adjusted_boxes.append(adjusted_box)
    print(str(count2) + " data created")
    count2 += 1

images = np.array(images)
adjusted_boxes = np.array(adjusted_boxes)
labels = np.array(labels)

# 학습 및 검증 데이터로 분리
X_train, X_val, y_train, y_val, boxes_train, boxes_val = train_test_split(images, labels, adjusted_boxes, test_size=0.2, random_state=42)

# YOLOv4 모델 불러오기
yolo = YOLOv4()

# 데이터셋 생성
train_dataset = YOLODataset(X_train, boxes_train, labels)
val_dataset = YOLODataset(X_val, boxes_val, y_val)

# 콜백 정의: 학습 중에 손실과 정확도를 출력
class PrintLossAndAccuracy(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1}/{self.params['epochs']} - loss: {logs['yolo_output_loss']} - accuracy: {logs['yolo_output_accuracy']} - val_loss: {logs['val_yolo_output_loss']} - val_accuracy: {logs['val_yolo_output_accuracy']}")

# 모델 학습
yolo.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[PrintLossAndAccuracy()])

# 예측을 수행할 때는 yolo.predict 대신에 model.predict를 사용합니다.
yolo.predict(X_val)
