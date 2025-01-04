import torch
from torch import nn

class Trainer(object):
    def __init__(self, args, model, dataloader=None, logger=None):
        super().__init__()
        self.model = model
        self.epochs = args.n_epoch
        self.dataloader = dataloader
        self.checkpoint_dir = args.checkpoint_dir
        self.multistep = args.multistep
        self.net = model.net
        
    def train_single(self,  **kwargs):
        loss_fn = nn.MSELoss()
        opt = torch.optim.Adam(self.net.parameters(),lr=1e-4)
        losses = []
        for epoch in range(self.epochs):
            count = 0
            for x in self.dataloader: 
                count += 1
                pred, noise = self.model(x)
                loss = loss_fn(pred, noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            avg_loss = sum(losses[-len(self.dataloader):])/len(self.dataloader)
            print(f'Finished epoch {epoch+1}. Average loss: {avg_loss:05f}')
            if (epoch+1) % 5 == 0 :
                model_file = f"{self.checkpoint_dir}/model_epoch_i{epoch+1}.pth"
                torch.save(self.net.state_dict(), model_file)
        return losses
    
    def train_multi(self, **kwargs):
        opt = torch.optim.Adam(self.net.parameters(),lr=1e-4)
        losses = []
        for epoch in range(self.epochs):
            count = 0
            for x in self.dataloader:
                if count == 61:
                    count += 1
                    continue
                loss = self.model(x,count,self.multistep)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
                count += 1 
            avg_loss = sum(losses[-len(self.dataloader):])/len(self.dataloader)
            print(f'Finished epoch {epoch+1}. Training Steps: {self.multistep}. Average loss: {avg_loss:05f}')
            model_file = f"{self.checkpoint_dir}/model_cc_epoch_{epoch+1}.pth"
            torch.save(self.net.state_dict(), model_file)
        return losses

    def sampling_single(self, **kwargs):
        real = None
        sample = None
        for x in self.dataloader:
            real, sample = self.model.Reverse_I(x,real,sample)
        return real,sample







        






        