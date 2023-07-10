import torch
import torch.nn as nn
import argparse
from torchvision import transforms
from tqdm import tqdm
import os
from utils.dataLoader import *
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
from torch.utils.data import random_split



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

lva_images = [image for image in train_images if 'LI-A' in image]
lva_images.sort()
# CSV 파일 로드
df = pd.read_csv('/home/minseopark/바탕화면/pipline/train.csv')
LVA_df = df[["L_ICA", "L_PCOM", "L_AntChor", "L_ACA","L_ACOM","L_MCA"]]

zero_row_indices = LVA_df[(LVA_df == 0).all(axis=1)].index.tolist()

lva_images = [image for i, image in enumerate(lva_images) if i not in zero_row_indices]

LVA_df = LVA_df[(LVA_df != 0).any(axis=1)].values.tolist()
# LVA_df["detected"] = 0
# LVA_df.loc[(LVA_df == 1).any(axis=1), "detected"] = 1


# 인덱스 열 제외하고 레이블 생성
# labels = data.drop('Index', axis=1).values.tolist()

# labels를 Tensor로 변환
labels_tensor = torch.tensor(LVA_df, dtype=torch.float32)

# 이미지 전처리를 위한 transform 설정
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # 정규화
])



train_dataset = CreateAngleDataset(
    lva_images, labels_tensor, transform=preprocess_transform)

# train_size = int(0.8 * len(train_dataset)) 
# val_size = len(train_dataset) - train_size  

# train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

batch_size = 16

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

# val_dataloader = DataLoader(
#     val_dataset, batch_size=batch_size, shuffle=True)


model = timm.create_model('coatnet_3_rw_224.sw_in12k', pretrained=False)

in_features = model.head.fc.in_features

# 새로운 FC 레이어 생성
new_fc = nn.Linear(in_features, 6)

# 모델의 마지막 FC 레이어를 새로운 FC 레이어로 대체
model.head.fc = new_fc
# state_dict = torch.load("/home/minseopark/바탕화면/pipline/coatnet_3_rw_224.sw_in12k_50.pt")
# model.load_state_dict(state_dict)
model.to("cuda")





device = torch.device("cuda")

for epoch in range(num_epochs):

    # 손실 초기화
    running_loss = 0.0

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for inputs, labels in tqdm(train_dataloader):
        # GPU를 사용하는 경우, 데이터를 GPU로 이동
        model.train()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        outputs = torch.squeeze(outputs)
        labels = torch.squeeze(labels)
        aneurysm_loss = criterion(torch.sigmoid(outputs[0]), labels[0])
        location_loss = criterion(torch.sigmoid(outputs[1:]), labels[1:])
        total_loss = aneurysm_loss + location_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
    epoch_loss = running_loss / len(train_dataloader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")

    

    if (epoch + 1) % 20 != 0:
        continue
    model.eval()
    with torch.no_grad():
        detect_predicted = []
        detect_true = []
        for inputs, labels in tqdm(train_dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = torch.squeeze(outputs)
            
            detect_predicted.extend(outputs.detach().cpu().numpy())
            detect_true.extend(labels.numpy())
        detect_predicted = np.array(detect_predicted)
        detect_true = np.array(detect_true)
        detect_predicted = np.where(detect_predicted > 0.5, 1, 0)
        auroc_train = roc_auc_score(detect_true, detect_predicted, multi_class='ovr')
        print(
            f"Epoch [{epoch+1}/{num_epochs}],AUROC_Train: {auroc_train}")
        detect_predicted = []
        detect_true = []
        # for inputs, labels in tqdm(val_dataloader):
        #     inputs = inputs.to(device)
        #     outputs = model(inputs)
        #     outputs = torch.squeeze(outputs)
            
        #     detect_predicted.extend(outputs.detach().cpu().numpy())
        #     detect_true.extend(labels.numpy())
        # detect_predicted = np.array(detect_predicted)
        # detect_true = np.array(detect_true)
        # detect_predicted = np.where(detect_predicted > 0.5, 1, 0)
        # auroc_train = roc_auc_score(detect_true, detect_predicted, multi_class='ovr')
        # print(
        #     f"Epoch [{epoch+1}/{num_epochs}],AUROC_Train: {auroc_train}")
        # torch.save(model.state_dict(),
        #         f"coatnet_3_rw_224.sw_in12k_{epoch}.pt")

# TODO ensemble model

# TODO inference

# TODO show result
