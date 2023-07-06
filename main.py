import torch
import torch.nn as nn
import argparse
from torchvision import transforms
from tqdm import tqdm
import os
from utils.dataLoader import CreateDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import timm
from torchsummary import summary
from torchsummary import summary
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import random


def parse_img_path(img_path):
    img_name = img_path.split('/')[-1]
    return img_name[0:4]

# TODO get all of argument
# parser = argparse.ArgumentParser(
#                     prog = 'inference',
#                     description = 'config of ',
#                     epilog = 'Text at the bottom of help')
# parser = argparse.ArgumentParser(description='Ensemble Model with Image Preprocessing')
# parser.add_argument('--ensemble', action='store_true', help='Use ensemble model')
# parser.add_argument('--checkpoint', action='store_true', help='Use model checkpoint')
# parser.add_argument('--preprocess', choices=['all', 'canny', 'transform'], default='all', help='Image preprocessing method')
# parser.add_argument('model_name', type=str, help='Model name')
# args = parser.parse_args()


# TODO Hyperparameter 초기화
num_epochs = 10000


# TODO get image from folder
# 데이터셋의 경로 설정
canny_train_path = './dataset/canny_train_set'
rotate_train_path = './dataset/canny_train_set_rotate'
train_path = './dataset/train_set'

# 경로에 있는 이미지 파일 가져오기
canny_train_images = [os.path.join(canny_train_path, filename)
                      for filename in os.listdir(canny_train_path)]
rotate_train_images = [os.path.join(
    rotate_train_path, filename) for filename in os.listdir(rotate_train_path)]
train_images = [os.path.join(train_path, filename)
                for filename in os.listdir(train_path)]
train_images.sort()
# CSV 파일 로드
data = pd.read_csv('/home/minseopark/바탕화면/pipline/train.csv')

# 인덱스 열 제외하고 레이블 생성
labels = data.drop('Index', axis=1).values.tolist()

# labels를 Tensor로 변환
labels_tensor = torch.tensor(labels, dtype=torch.float32)


# 이미지 전처리를 위한 transform 설정
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # 정규화
])

# aneurysm label이 0인 데이터와 1인 데이터 분리
label_data = data.values.tolist()   # index, labels 모두 포함된 리스트
label_data_0 = []   # aneurysm label이 0인 것만 뽑아내기
image_data_0 = []
label_data_1 = []   # aneurysm label이 1인 것만 뽑아내기
image_data_1 = []

image_dict = {}
count = 0
# image dictionary 만들기 (편의상)
for img in train_images:
    if count == 0:
        image_dict[parse_img_path(img)] = [img]
        count += 1
    else:
        image_dict[parse_img_path(img)].append(img)
        count += 1
        if count == 8:
            count = 0

# label 0, 1 분류하기
for ld in label_data:
    idx = ld[0]

    if ld[1] == 0:
        label_data_0.append(ld)
        image_data_0 = image_data_0 + image_dict[str(idx)]
    else:
        label_data_1.append(ld)
        image_data_1 = image_data_1 + image_dict[str(idx)]

# 두 리스트 길이의 합이 기존 리스트와 같은지 확인
if len(label_data_0) + len(label_data_1) != len(label_data):
    exit()
else:
    print(f"aneurysm label이 0인 label 데이터 개수 : {len(label_data_0)}")
    print(f"aneurysm label이 0인 image 데이터 개수 : {len(image_data_0)}")
    print(label_data_0)
    print(image_data_0)
    print(f"aneurysm label이 1인 label 데이터 개수 : {len(label_data_1)}")
    print(f"aneurysm label이 1인 image 데이터 개수 : {len(image_data_1)}")
    print(label_data_1)
    print(image_data_1)

# Training data와 Validation data 분리
train_label_0, val_label_0 = train_test_split(
    label_data_0, test_size=0.2, random_state=42)
train_label_1, val_label_1 = train_test_split(
    label_data_1, test_size=0.2, random_state=42)

train_labels = train_label_0 + train_label_1
val_labels = val_label_0 + val_label_1
random.seed(42)   # randomization results are always the same for each run
random.shuffle(train_labels)
random.shuffle(val_labels)

train_images = []
val_images = []
for l in train_labels:
    train_images = train_images + image_dict[str(l[0])]
for l in val_labels:
    val_images = val_images + image_dict[str(l[0])]

print(train_labels)
train_labels = [sublist[1:] for sublist in train_labels]
print(train_images)
print(val_labels)
val_labels = [sublist[1:] for sublist in val_labels]
print(val_images)

# labels를 Tensor로 변환
train_labels = torch.tensor(train_labels, dtype=torch.float32)
val_labels = torch.tensor(val_labels, dtype=torch.float32)

# 이미지 경로를 데이터셋으로 변환
# canny_train_dataset = CreateDataset(canny_train_images, transform=preprocess_transform)
# rotate_train_dataset = CreateDataset(rotate_train_images, transform=preprocess_transform)
train_dataset = CreateDataset(
    train_images, train_labels, transform=preprocess_transform)
val_dataset = CreateDataset(val_images, val_labels,
                            transform=preprocess_transform)

batch_size = 32
# canny_train_dataloader = DataLoader(canny_train_dataset, batch_size=batch_size, shuffle=True)
# rotate_train_dataloader = DataLoader(rotate_train_dataset, batch_size=batch_size, shuffle=True)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# TODO transformation and gan image creation
# TODO transformation and gan image creation

# TODO import model from file

model = timm.create_model('coatnet_3_rw_224.sw_in12k', pretrained=False)
in_features = model.head.fc.in_features

# 새로운 FC 레이어 생성
new_fc = nn.Linear(in_features, 22)

# 모델의 마지막 FC 레이어를 새로운 FC 레이어로 대체
model.head.fc = new_fc
model.to("cuda")
# 모델 구조 출력
summary(model, input_size=(3, 224, 224))
device = torch.device("cuda")

for epoch in range(num_epochs):

    # 손실 초기화
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for inputs, labels in tqdm(train_dataloader):
        # GPU를 사용하는 경우, 데이터를 GPU로 이동
        model.train()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        outputs = torch.squeeze(outputs)
        labels = torch.squeeze(labels)
        # 추가적인 처리 작업 수행
        loss = criterion(outputs, labels)
        # Backward pass: 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")

    model.eval()
    y_true = []
    y_pred = []
    y_pred_prob = []
    epoch = 0
    if epoch % 100 != 0:
        continue
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(val_dataloader):
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.tolist()
            y_pred_prob.extend(outputs)
            outputs = [[0 if i < 0.5 else 1 for i in results]
                       for results in outputs]
            y_true.extend(batch_labels.tolist())
            y_pred.extend(outputs)

    y_true = np.array(y_true).astype(int).tolist()

    # aneurysm 여부에 대한 auroc 값만 계산하기
    aneurysm_true = [sublist[0] for sublist in y_true]
    aneurysm_pred = [sublist[0] for sublist in y_pred_prob]
    auroc = roc_auc_score(aneurysm_true, aneurysm_pred)

    # 위치 정보에 대한 accuracy 계산하기
    location_true = [sublist[1:] for sublist in y_true]
    location_pred = [sublist[1:] for sublist in y_pred]
    location_true = torch.tensor(
        [item for sublist in location_true for item in sublist])
    location_pred = torch.tensor(
        [item for sublist in location_pred for item in sublist])
    accuracy = (location_pred == location_true).sum().item() / \
        len(location_true)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], AUROC: {auroc}, Accuracy: {accuracy}")
    torch.save(model.state_dict(),
               f"/content/drive/MyDrive/medai/models/resnet50s_{epoch}.pt")

# TODO ensemble model

# TODO inference

# TODO show result
