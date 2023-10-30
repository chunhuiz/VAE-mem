import torch
from torch import nn

def load_c3d_pretrained_model(net,checkpoint_path,name=None):
    checkpoint = torch.load(checkpoint_path)
    print(checkpoint.keys())
    state_dict = net.state_dict()
    print(state_dict.keys())
    base_dict = {}
    checkpoint_keys = checkpoint.keys()
    if name==None:
        for k, v in state_dict.items():
            for _k in checkpoint_keys:

                if k in _k:
                    print(k)
                    base_dict[k] = checkpoint[_k]
    else:
        if name=='fc6':
            base_dict['0.weight']=checkpoint['backbone.fc6.weight']
    #         base_dict['0.bias']=checkpoint['backbone.fc6.bias']
    # import pdb
    # pdb.set_trace()
    state_dict.update(base_dict)
    net.load_state_dict(state_dict)
    print('model load pretrained weights')
    
class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        # 112
        self.conv1a = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # 56
        self.conv2a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 28
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 14
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 7
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.GAP=nn.AdaptiveAvgPool3d(1)

        # self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        # self.pool5=nn.AdaptiveAvgPool3d(1)
        # self.fc6 = nn.Linear(8192, 4096)

        self.relu = nn.ReLU()

    def forward(self,x):

        x = self.relu(self.conv1a(x))
        x = self.pool1(x)
        x = self.relu(self.conv2a(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        out_4=x

        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))

        feat=self.GAP(x).squeeze(-1).squeeze(-1).squeeze(-1)

        # x = self.pool5(x)

        # x = x.view(-1, 8192)
        # x = self.relu(self.fc6(x))
        return feat
    
    
