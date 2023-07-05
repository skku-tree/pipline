import torch
import torch.nn as nn
import torch.optim as optim
from models.coatnet import coatnet_0

class Inference:
    def __init__(self,model,args,config):
        self.model = model
        self.args= args
        self.config= config

    def train(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # Import the dataset
        dataset = CountingkDataset(self.config,'train')
        data_loader = DataLoader(dataset,self.config.train.batch_size,shuffle=True,pin_memory=True,num_workers=self.config.train.num_workers)
        print("dataset length ", len(dataset))

        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(),lr=self.config.optimizer.lr)

        for epoch in range(self.config.train.epochs):
            for i, sample in enumerate(data_loader):
                optimizer.zero_grad()
                
                queries = sample[0].to(self.config.device)
                references = sample[1].to(self.config.device)
                target = sample[2].to(self.config.device)

                self.model.train()
                FS = self.model(queries,references)
                loss, ssim_loss = self.model(FS,target,self.config)
                
                loss.backward()
                optimizer.step()
                
                print(f"Loss: {loss / self.config.train.batch_size}, SSIM Loss: {ssim_loss / self.config.train.batch_size}")

    def test(self):
        
        net = CFOCNet()
        
        net.to(self.config.device)

        checkpoint = torch.load(self.config.eval.checkpoint)
        net.load_state_dict(checkpoint)
        net.eval()

       
        dataset = CountingkDataset(self.config,'train',[3])
        data_loader = DataLoader(dataset,self.config.train.batch_size,pin_memory=True,num_workers=self.config.train.num_workers)

        mae_sum = 0
        mse_sum = 0
        
        count = len(dataset)

        with torch.no_grad():
            for i, sample in enumerate(data_loader):

                queries = sample[0].to(self.config.device)
                references = sample[1].to(self.config.device)
                target = sample[2].to(self.config.device)

                predict = net(queries,references)

                logging.info(f'target num: {torch.sum(target).item()}, predict num: {torch.sum(predict).item()}')

                target_num = torch.sum(target).item()
                predict_num = torch.sum(predict).item()

                mae_sum += abs(target_num-predict_num)
                mse_sum += abs(target_num-predict_num) **2