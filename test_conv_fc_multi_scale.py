import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("./training_data_for_super_resolution_of_position.txt")
force = data[:,1:7]
sensor = data[:,7:43]
pos = data[:,-3:-1]
print("force : {}, sensor: {}, pos.shape: {}".format(force.shape, sensor.shape, pos.shape))

dy = [3.3, 3.3, 2.4, 3.3, 3.3, 2.4, 3.3, 3.3]
point_pos = np.zeros((2,9,9))
for i in range(9):
    point_pos[1, :, i] = 27.3 - sum(dy[:i])
    point_pos[0, i, :] = -1.5 + sum(dy[:i])

def sensor2img(sensor_data):
    # 1x36 to 9x9 (corresponding to the actual position of each sensor)
    feature = np.zeros((sensor_data.shape[0], 3, 9, 9))
    idx_f1 = [1,4,7]
    idx_f2 = [0,2,3,5,6,8]
    idx_feature = [idx_f1, idx_f2, idx_f1, idx_f1, idx_f2, idx_f1, idx_f1, idx_f2, idx_f1]
    idx_sensor = [[35,15,11],[36,34,16,14,12,10],[33,13,9],[31,19,7],[32,30,20,18,8,6],[29,17,5],[27,23,3],[28,26,24,22,4,2],[25,21,1]]
    for i in range(9):
        for j in range(len(idx_feature[i])):
            feature[:, 0, i, idx_feature[i][j]] = sensor_data[:,idx_sensor[i][j]-1]
            feature[:, 1:, :, :] = point_pos
    return feature

feature = sensor2img(sensor)


#TODO: design a nn that takes feature (whose shape is (9,9)) as input, and outputs a heatmap (whose shape is (9,9))
#Version1: CNN stem + attention (from HaloNet: "Scaling Local Self-Attention For Parameter Efficient Visual Backbones", CVPR, 2021 (Google).)
import torch
from torch import nn, einsum
import torch.nn.functional as F
# extend the dimension of feature map
class sensor2Heatmap(nn.Module):
    def __init__(self, point_pos, device):
        super(sensor2Heatmap, self).__init__()
        self.point_pos = torch.from_numpy(point_pos).to(device).permute(2,0,1)
        
        self.ds_conv_p1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,5,1,1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),

            torch.nn.Conv2d(16,32,5,1,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(), 

            torch.nn.Conv2d(32,64,5,1,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        ) #(N, 64, 3, 3)
        self.ds_conv_p2 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,5,1,2),
            torch.nn.ReLU(), 
            torch.nn.Conv2d(64,128,5,1,2),
            torch.nn.ReLU(), 
            torch.nn.Conv2d(128,256,5,1,1),
        ) #(N, 512, 1, 1)

        self.cm_conv_p1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,5,1,2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(), 
            torch.nn.Conv2d(16,32,5,1,2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(), 
            torch.nn.Conv2d(32,64,5,1,2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        ) #(N, 64, 9, 9)
        self.cm_conv_p2 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,5,1,1),
            torch.nn.ReLU(), 
            torch.nn.Conv2d(64,128,5,1,1),
            torch.nn.ReLU(), 
            torch.nn.Conv2d(128,256,5,1,0),
        ) #(N, 512, 1, 1)

        self.us_conv_p1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,5,1,3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(), 
            torch.nn.Conv2d(16,32,5,1,3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(), 
            torch.nn.Conv2d(32,64,5,1,3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        ) #(N, 64, 15, 15)
        self.us_conv_p2 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,9,1,1),
            torch.nn.ReLU(), 
            torch.nn.Conv2d(64,128,9,1,1),
            torch.nn.ReLU(), 
            torch.nn.Conv2d(128,256,5,1,1),
        ) #(N, 512, 1, 1)    

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256*3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    
    def forward(self, x):
        ds_x = self.ds_conv_p1(x)
        cm_x = self.cm_conv_p1(x)
        us_x = self.us_conv_p1(x)
        
        ds_pos = self.ds_conv_p2(ds_x)
        cm_pos = self.cm_conv_p2(cm_x)
        us_pos = self.us_conv_p2(us_x)

        ds_pos_flatten = torch.flatten(ds_pos, start_dim=1)
        cm_pos_flatten = torch.flatten(cm_pos, start_dim=1)
        us_pos_flatten = torch.flatten(us_pos, start_dim=1)

        fc_pos = torch.cat([ds_pos_flatten, cm_pos_flatten, us_pos_flatten], dim=-1)

        output = self.fc(fc_pos)
        return output.float()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Model runs on {}.".format(device))
encoder_x = sensor2Heatmap(point_pos, device).to(device)
#encoder_y = sensor2Heatmap(point_pos, device).to(device)
encoder_x.load_state_dict(torch.load("/home/zhy/project/sensor_super_resolution/2023-09-07/2023-09-07-x_fc_multiscale_512_running_best.pth"))
#encoder_y.load_state_dict(torch.load("/home/zhy/project/sensor_super_resolution/2023-09-07/2023-09-07-y_fc_multiscale_512_running_best.pth"))
encoder_x.train()
#encoder_y.train()

import torch.utils.data as Data
train_x = torch.from_numpy(feature).float()
train_y = torch.from_numpy(pos).float()
torch_dataset = Data.TensorDataset(train_x, train_y)

loader = Data.DataLoader(
    dataset=torch_dataset,      
    batch_size=40000, 
    shuffle=True,
)


loss_fun = torch.nn.MSELoss()
optx = torch.optim.Adam(encoder_x.parameters(), lr=1e-4)
#opty = torch.optim.Adam(encoder_y.parameters(), lr=1e-4)

best_lossx = 0.08
#best_lossy = 1e10

encoder_x = encoder_x.to(device)
#encoder_y = encoder_y.to(device)

for epoch in range(1000):
    k, train_lossx, train_lossy = 0, 0, 0
    for i, (batch_x, batch_y) in enumerate(loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        noise = torch.rand(batch_x.shape) * 0.0001
        noise = noise.to(device)
        outputx = encoder_x(batch_x+noise).squeeze()
        #outputy = encoder_y(batch_x+noise).squeeze()
        loss_x = loss_fun(outputx, batch_y[:,0].squeeze())
        #loss_y = loss_fun(outputy, batch_y[:,1].squeeze())
        train_lossx += loss_x.item()
        #train_lossy += loss_y.item()
        optx.zero_grad()
        #opty.zero_grad()
        loss_x.backward()
        #loss_y.backward()
        optx.step()
        #opty.step()
        k += 1

    train_lossx /= k
    #train_lossy /= k
    
    if train_lossx < best_lossx:
        torch.save(encoder_x.state_dict(), "./2023-09-07-x_fc_multiscale_512_running_best.pth")
        best_lossx = train_lossx
    #if train_lossy < best_lossy:
    #    torch.save(encoder_y.state_dict(), "./2023-09-07-y_fc_multiscale_512_running_best.pth")
    #    best_lossy = train_lossy
    print("Epoch: {}, X RMSE Loss: {}".format(epoch, np.sqrt(train_lossx)))


