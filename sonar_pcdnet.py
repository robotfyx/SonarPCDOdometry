import torch
import torch.nn as nn
from SonarPCDNet.pcdnet_modules import *
from matplotlib import pyplot as plt

class SonarPCDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = 64
        self.n2 = 128
        self.n3 = 256

        ### Siamese Point Feature Pyramid ###
        self.psa1 = SAModule(npoint=1024, nsample=32, narc=10, m=6, mlp=[0, 64, 64, self.n1], bn=True)
        self.psa2 = SAModule(npoint=512, nsample=16, narc=6, m=3, mlp=[64, 128, 128, self.n2], bn=True)
        self.psa3 = SAModule(npoint=256, nsample=16, narc=3, m=1, mlp=[128, 256, 256, self.n3], bn=True)
        # self.psa4 = SAModule(npoint=64, nsample=16, mlp=[512, 512, 512, 512], bn=False)

        ### Attentive Cost Volume ###
        self.cost_volume3 = CostVolume(nsample=4, nsample_q=16, in_channel1=self.n3, in_channel2=self.n3, mlp1=[128, 128, self.n3], mlp2=[128, self.n3], m=1)
        self.cost_volume2 = CostVolume(nsample=8, nsample_q=16, in_channel1=self.n2, in_channel2=self.n2, mlp1=[64, 128, self.n2], mlp2=[128, self.n2], m=3)
        self.cost_volume1 = CostVolume(nsample=16, nsample_q=32, in_channel1=self.n1, in_channel2=self.n1, mlp1=[64, 64, self.n1], mlp2=[128, self.n1], m=6)

        ### Occupancy Probability Mask ###
        self.Mask3 = MaskPredictor(in_channel=self.n3*2, mlp=[128,self.n3])
        self.Mask2 = MaskPredictor(in_channel=self.n2*2, mlp=[64,self.n2])
        self.Mask1 = MaskPredictor(in_channel=self.n1*2, mlp=[32,self.n1])

        ### Pose Predictor ###
        # self.PosePredictor = PosePredictor(in_channel=64, out_channel=256)

        ### Flow Feature Encoding ###
        # self.flow_feature_encoding = SAModule(npoint=64, nsample=16, mlp=[512, 1024, 1024, 512], bn=False)
        ### PCD Predictor ###
        self.PCD_Predictor = RecoverPCD(num_channel=self.n3+self.n2+self.n1)
        self.narc = 10
        
    def forward(self, xyz_f1:torch.Tensor, features_f1:torch.Tensor, xyz_f2:torch.Tensor, features_f2:torch.Tensor, nout:int, rtheta:torch.Tensor):
        '''
        input:
        xyz_f1: [B, N, 3]
        features_f1: [B, N, C1]
        xyz_f2: [B, N, 3]
        features_f2: [B, N, C2]

        rtheta: [B, numpoints, 2]

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

        # fig = plt.figure()
        # ax1 = fig.add_subplot(221, projection='3d')
        # ax1.scatter(xyz_f1.detach().cpu().numpy()[0, :, 0], xyz_f1.detach().cpu().numpy()[0, :, 1], xyz_f1.detach().cpu().numpy()[0, :, 2], c='r')
        # ax2 = fig.add_subplot(222, projection='3d')
        # ax2.scatter(new_xyz_f1_1.detach().cpu().numpy()[0, :, 0], new_xyz_f1_1.detach().cpu().numpy()[0, :, 1], new_xyz_f1_1.detach().cpu().numpy()[0, :, 2], c='r')
        # ax3 = fig.add_subplot(223, projection='3d')
        # ax3.scatter(new_xyz_f1_2.detach().cpu().numpy()[0, :, 0], new_xyz_f1_2.detach().cpu().numpy()[0, :, 1], new_xyz_f1_2.detach().cpu().numpy()[0, :, 2], c='r')
        # ax4 = fig.add_subplot(224, projection='3d')
        # ax4.scatter(new_xyz_f1_3.detach().cpu().numpy()[0, :50, 0], new_xyz_f1_3.detach().cpu().numpy()[0, :50, 1], new_xyz_f1_3.detach().cpu().numpy()[0, :50, 2], c='r')
        # ax4.scatter(new_xyz_f2_3.detach().cpu().numpy()[0, :50, 0], new_xyz_f2_3.detach().cpu().numpy()[0, :50, 1], new_xyz_f2_3.detach().cpu().numpy()[0, :50, 2], c='b')
        # ax1.set_aspect('equal')
        # ax2.set_aspect('equal')
        # ax3.set_aspect('equal')
        # ax4.set_aspect('equal')
        # ax1.set_xlabel('x')
        # ax2.set_xlabel('x')
        # ax3.set_xlabel('x')
        # ax4.set_xlabel('x')

        # ax1.set_ylabel('y')
        # ax2.set_ylabel('y')
        # ax3.set_ylabel('y')
        # ax4.set_ylabel('y')
        # plt.show()

        new_xyz_f1_3_t = torch.permute(new_xyz_f1_3, (0, 2, 1)).contiguous() # [B, 3, 256]
        new_xyz_f2_3_t = torch.permute(new_xyz_f2_3, (0, 2, 1)).contiguous() # [B, 3, 256]
        new_xyz_f1_2_t = torch.permute(new_xyz_f1_2, (0, 2, 1)).contiguous() # [B, 3, 256]
        new_xyz_f2_2_t = torch.permute(new_xyz_f2_2, (0, 2, 1)).contiguous() # [B, 3, 256]
        new_xyz_f1_1_t = torch.permute(new_xyz_f1_1, (0, 2, 1)).contiguous() # [B, 3, 256]
        new_xyz_f2_1_t = torch.permute(new_xyz_f2_1, (0, 2, 1)).contiguous() # [B, 3, 256]

        ### Attentive Cost Volume ###
        cost_volume3 = self.cost_volume3(new_xyz_f1_3_t, new_features_f1_3_t, new_xyz_f2_3_t, new_features_f2_3_t) # [B, 512, 256]
        cost_volume2 = self.cost_volume2(new_xyz_f1_2_t, new_features_f1_2_t, new_xyz_f2_2_t, new_features_f2_2_t)
        cost_volume1 = self.cost_volume1(new_xyz_f1_1_t, new_features_f1_1_t, new_xyz_f2_1_t, new_features_f2_1_t)

        # ### Flow Feature Encoding ###
        # # _, embedding_features = self.flow_feature_encoding(new_xyz_f1_3, flow_embedding) # [B, 64, 3] [B, 512, 64]
        # mask3 = self.Mask3(new_features_f1_3_t, cost_volume3) # [B, 512, 256]
        # mask3 = F.softmax(mask3, dim=2)
        # mask2 = self.Mask2(new_features_f1_2_t, cost_volume2) # [B, 512, 256]
        # mask2 = F.softmax(mask2, dim=2)
        # mask1 = self.Mask1(new_features_f1_1_t, cost_volume1) # [B, 512, 256]
        # mask1 = F.softmax(mask1, dim=2)

        # embedding3 = torch.sum(cost_volume3*mask3, dim=2, keepdim=True)
        # embedding2 = torch.sum(cost_volume2*mask2, dim=2, keepdim=True)
        # embedding1 = torch.sum(cost_volume1*mask1, dim=2, keepdim=True)
        embedding3 = torch.mean(cost_volume3, dim=2, keepdim=True)
        embedding2 = torch.mean(cost_volume2, dim=2, keepdim=True)
        embedding1 = torch.mean(cost_volume1, dim=2, keepdim=True)
        embedding = torch.cat((embedding3, embedding2, embedding1), dim=1)
        # embedding = torch.cat((cost_volume3, cost_volume2, cost_volume1), dim=1)

        ### Pose Predictor ###
        # rv, t, points_out = self.PosePredictor(cost_volume, mask, xyz_f1.shape[1]/10)
        points_out = self.PCD_Predictor(embedding, nout, rtheta)
        # weights = self.PCD_Predictor(embedding, nout, rtheta) # [B, nout, narc]
        # # weights_one_hot = F.softmax(weights/1e-3, dim=2)
        # B, _, _ = xyz_f1.shape
        # xyz_f1_reshape = xyz_f1.view((B, nout, self.narc, 3))
        # weighted_points = (weights.unsqueeze(-1)*xyz_f1_reshape).sum(dim=2)
        
        return points_out #weighted_points, weights #

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
