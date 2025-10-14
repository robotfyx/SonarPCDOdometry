import torch
import torch.nn as nn
from SonarPCDNet.pcdnet_modules import *
import math

class SonarPCDNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.pose = pose

        ### Siamese Point Feature Pyramid ###
        self.psa1 = SAModule(npoint=2048, nsample=32, mlp=[0, 8, 8, 16], bn=True)
        self.psa2 = SAModule(npoint=1024, nsample=32, mlp=[16, 16, 16, 32], bn=True)
        self.psa3 = SAModule(npoint=256, nsample=16, mlp=[32, 32, 32, 64], bn=True)
        # self.psa4 = SAModule(npoint=64, nsample=16, mlp=[512, 512, 512, 512], bn=False)

        ### Attentive Cost Volume ###
        self.cost_volume = CostVolume(nsample=4, nsample_q=32, in_channel1=64, in_channel2=64, mlp1=[128, 64, 64], mlp2=[128, 64])

        ### Occupancy Probability Mask ###
        self.Mask = MaskPredictor(in_channel=64+64, mlp=[128,64])

        ### Pose Predictor ###
        # self.PosePredictor = PosePredictor(in_channel=64, out_channel=256)

        ### Flow Feature Encoding ###
        # self.flow_feature_encoding = SAModule(npoint=64, nsample=16, mlp=[512, 1024, 1024, 512], bn=False)
        ### PCD Predictor ###
        self.PCD_Predictor = RecoverPCD(in_channel=64, out_channel=256)
        
    def forward(self, xyz_f1:torch.Tensor, features_f1:torch.Tensor, xyz_f2:torch.Tensor, features_f2:torch.Tensor, nout:int):
        '''
        input:
        xyz_f1: [B, N, 3]
        features_f1: [B, N, C1]
        xyz_f2: [B, N, 3]
        features_f2: [B, N, C2]

        return:
        poses: 4d for quaternions and 3d for translation
        '''
        features_f1_t = torch.permute(features_f1, (0, 2, 1)).contiguous() # [B, C1, N]
        features_f2_t = torch.permute(features_f2, (0, 2, 1)).contiguous() # [B, C2, N]
        
        ### Set Abstraction ###
        new_xyz_f1_1, new_features_f1_1_t = self.psa1(xyz_f1, features_f1_t) # [B, 2048, 3] [B, 64, 2048]
        new_xyz_f1_2, new_features_f1_2_t = self.psa2(new_xyz_f1_1, new_features_f1_1_t) # [B, 1024, 3] [B, 256, 1024]
        new_xyz_f1_3, new_features_f1_3_t = self.psa3(new_xyz_f1_2, new_features_f1_2_t) # [B, 256, 3] [B, 512, 256]
        # new_xyz_f1_4, new_features_f1_4_t = self.psa4(new_xyz_f1_3, new_features_f1_3_t) # [B, 64, 3] [B, 512, 64]

        new_xyz_f2_1, new_features_f2_1_t = self.psa1(xyz_f2, features_f2_t) # [B, 2048, 3] [B, 64, 2048]
        new_xyz_f2_2, new_features_f2_2_t = self.psa2(new_xyz_f2_1, new_features_f2_1_t) # [B, 1024, 3] [B, 256, 1024]
        new_xyz_f2_3, new_features_f2_3_t = self.psa3(new_xyz_f2_2, new_features_f2_2_t) # [B, 256, 3] [B, 512, 256]
        # new_xyz_f2_4, new_features_f2_4_t = self.psa4(new_xyz_f2_3, new_features_f2_3_t) # [B, 64, 3] [B, 1024, 64]

        new_xyz_f1_3_t = torch.permute(new_xyz_f1_3, (0, 2, 1)).contiguous() # [B, 3, 256]
        new_xyz_f2_3_t = torch.permute(new_xyz_f2_3, (0, 2, 1)).contiguous() # [B, 3, 256]

        ### Attentive Cost Volume ###
        cost_volume = self.cost_volume(new_xyz_f1_3_t, new_features_f1_3_t, new_xyz_f2_3_t, new_features_f2_3_t) # [B, 512, 256]

        ### Flow Feature Encoding ###
        # _, embedding_features = self.flow_feature_encoding(new_xyz_f1_3, flow_embedding) # [B, 64, 3] [B, 512, 64]
        mask = self.Mask(new_features_f1_3_t, cost_volume) # [B, 512, 256]
        mask = F.softmax(mask, dim=2)

        ### Pose Predictor ###
        # rv, t, points_out = self.PosePredictor(cost_volume, mask, xyz_f1.shape[1]/10)
        points_out = self.PCD_Predictor(cost_volume, mask, nout)

        return points_out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SonarPCDNet()
    net = net.to(device)

    xyz_f1 = torch.rand((2, 3000, 3)).to(device)
    xyz_f2 = torch.rand((2, 3000, 3)).to(device)

    features_f1 = torch.rand((2, 3000, 3)).to(device)
    features_f2 = torch.rand((2, 3000, 3)).to(device)

    q, t, pts = net(xyz_f1, xyz_f2, features_f1, features_f2)
    print(q, t, pts.shape)
