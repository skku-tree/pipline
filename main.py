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
from sklearn.metrics import roc_auc_score
import timm
from torchsummary import summary

#TODO get all of argument
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


#TODO Hyperparameter 초기화
num_epochs = 10000


#TODO get image from folder
# 데이터셋의 경로 설정
canny_train_path = './dataset/canny_train_set'
rotate_train_path = './dataset/canny_train_set_rotate'
train_path = './dataset/train_set'

# 경로에 있는 이미지 파일 가져오기
canny_train_images = [os.path.join(canny_train_path, filename) for filename in os.listdir(canny_train_path)]
rotate_train_images = [os.path.join(rotate_train_path, filename) for filename in os.listdir(rotate_train_path)]
train_images = [os.path.join(train_path, filename) for filename in os.listdir(train_path)]
train_images.sort()
# CSV 파일 로드
data = pd.read_csv('/home/minseopark/바탕화면/pipline/train.csv')

# 인덱스 열 제외하고 레이블 생성
labels = data.drop('Index', axis=1).values.tolist()

# labels를 Tensor로 변환
labels_tensor = torch.tensor(labels,dtype=torch.float32)


# 이미지 전처리를 위한 transform 설정
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# 이미지 경로를 데이터셋으로 변환
# canny_train_dataset = CreateDataset(canny_train_images, transform=preprocess_transform)
# rotate_train_dataset = CreateDataset(rotate_train_images, transform=preprocess_transform)
train_dataset = CreateDataset(train_images,labels_tensor, transform=preprocess_transform)

batch_size = 16
# canny_train_dataloader = DataLoader(canny_train_dataset, batch_size=batch_size, shuffle=True)
# rotate_train_dataloader = DataLoader(rotate_train_dataset, batch_size=batch_size, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

#TODO transformation and gan image creation

#TODO import model from file

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
    if epoch % 100 != 0:
        continue
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(train_dataloader):
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.tolist()
            outputs = [[0 if i < 0.5 else 1 for i in results] for results in outputs]
            y_true.extend(batch_labels.tolist())
            y_pred.extend(outputs)
    auroc = roc_auc_score(y_true, y_pred)
    print(f"Epoch [{epoch+1}/{num_epochs}], AUROC: {auroc}")
    torch.save(model.state_dict(), f"coatnet_3_rw_224.sw_in12k_{epoch}.pt")
#TODO ensemble model

#TODO inference

#TODO show result 