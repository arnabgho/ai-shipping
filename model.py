import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
import torch.utils.data as data
pd_data = pd.read_csv('final_data.csv',delimiter=';')

forward_map = {}
backward_map = {}
maxim_target =  pd_data['Dt'].values.max()
pd_data['Dt']=pd_data['Dt']/maxim_target
i=0
for val in pd.unique(pd_data['MMSI']):
    forward_map[i]=val
    backward_map[val]=i
    i+=1


embed_size = 10
num_hidden = 256
batch_size = 1024
lr = 1e-3
niter = 100
outf = './'
class ship(nn.Module):
    def __init__(self,embed_size=10,num_hidden=256):
        super(ship,self).__init__()
        self.model = nn.Sequential(nn.Linear(embed_size+15,num_hidden),
                      nn.ReLU(),
                      nn.Linear(num_hidden,num_hidden),
                      nn.ReLU(),
                      nn.Linear(num_hidden,int(num_hidden/2)),
                      nn.ReLU(),
                      nn.Linear(int(num_hidden/2),int(num_hidden/4)),
                      nn.ReLU(),
                      nn.Linear(int(num_hidden/4),1)
                      )
        self.embed_layer = nn.Embedding(417,embed_size)

    def forward(self,x,ship_id):
        embedded_vec = self.embed_layer(ship_id)
        embedded_vec= embedded_vec.squeeze(1)
        x =torch.cat((embedded_vec,x),1)
        return self.model(x)


class ShippingDataset(data.Dataset):
    def __init__(self,pandas_dataset):
        self.pandas_dataset = pandas_dataset

    def __getitem__(self,index):
        target = torch.Tensor([self.pandas_dataset['Dt'].values[index]])
        shipping_id = torch.LongTensor([backward_map[self.pandas_dataset['MMSI'].values[index]]])
        covariates = ['Latitude', 'Longitude', 'SOG', 'Width', 'Length', 'Draught',
                   'Dx_long', 'Dx_lat', 'min_rain', 'max_rain', 'wave_direction',
                          'wave_height', 'wave_period', 'wind_east', 'wind_north']
        a = []
        for covariate in covariates:
            a.append(self.pandas_dataset[covariate].values[index])

        covariate_tensor = torch.Tensor(a)
        #print(covariate_tensor.size())
        #print(target.size())
        #print(target.size())
        #return target,shipping_id,covariate_tensor
        return { "Target":target, "Shipping_ID":shipping_id, "Covariate":covariate_tensor }


    def __len__(self):
        return len(self.pandas_dataset['Dt'].values)

ai_shipping_model = ship(embed_size,num_hidden)
ai_shipping_model = ai_shipping_model.cuda()
optimizer = optim.Adam(ai_shipping_model.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.MSELoss()
criterion = criterion.cuda()


dataset = ShippingDataset(pd_data)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=1)

for epoch in range(niter):
    for i, data in enumerate(dataloader, 0):
        ai_shipping_model.zero_grad()
        covariates =data['Covariate'].cuda()
        shipping_ids = data['Shipping_ID'].cuda()
        targets = data['Target'].cuda()

        #covariates =data[0].cuda()
        #shipping_ids = data[1].cuda()
        #targets = data[2].cuda()


        predictions = ai_shipping_model(covariates,shipping_ids)

        err = criterion(predictions,targets)

        print('[%d/%d][%d/%d] Loss_D: %.4f'%(epoch,niter,i,len(dataloader),err.item()))

    torch.save(ai_shipping_model.state_dict(), '%s/ai_shipping_model_epoch_%d.pth' % (outf, epoch))


