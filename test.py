import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sonar_pcdnet import SonarPCDNet
from Sonarpcd_data import SonarPCDData
import os
from utils import get_dataloader_workers, pad, sinkhorn_emd, rotvec2mat
from matplotlib import pyplot as plt
from transforms3d import quaternions
import numpy as np

if __name__ == '__main__':
    basedir = '/media/kemove/043E0D933E0D7F44/Sonar_Diffusion_SLAM'
    data = SonarPCDData(os.path.join(basedir, 'data'))
    dataloader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=get_dataloader_workers()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SonarPCDNet().to(device)
    checkpoints = torch.load(os.path.join(basedir, 'trainlog/exp4/model_5.pth'), map_location=device)
    model.load_state_dict(checkpoints["model"])
    model.eval()

    for i, (pcd1, pcd2, f1, f2, R, t, pts1_gt, pts2_gt) in enumerate(dataloader):
        with torch.no_grad():
            pcd1 = pcd1.to(device).float()
            pcd2 = pcd2.to(device).float()
            f1 = f1.to(device).float()
            f2 = f2.to(device).float()

            R = R.to(device).float()
            t = t.to(device).float()
            pts1_gt = pts1_gt.to(device).float()
            pts2_gt = pts2_gt.to(device).float()

            pts1_pre, weights1 = model(pcd1, f1, pcd2, f2, pts1_gt.shape[1], f1[:, ::10, :2])
            xyz = pcd1.view((1, pts1_gt.shape[1], 10, 3))
            print(weights1[0, 0], xyz[0, 0])
            # weights = nn.functional.softmax(weights1/1e-3, dim=2)
            # print(weights[0, 0])
            pts2_pre, weights2 = model(pcd2, f2, pcd1, f1, pts2_gt.shape[1], f2[:, ::10, :2])

            q1 = pts1_pre-torch.mean(pts1_pre, dim=1)
            q2 = pts2_pre-torch.mean(pts2_pre, dim=1)
            # q2 = pts2_gt-torch.mean(pts2_gt, dim=1)
            W = torch.einsum('bni,bnj->bij', q2, q1).squeeze(0)
            U, S, V = torch.linalg.svd(W)

            # R_pre = (U @ V).t()
            det = torch.linalg.det((U @ V).t())
            D = torch.eye(3, device=device)
            D[2, 2] = torch.sign(det)
            R_pre = V.t() @ D @ U.t()
                
            t_pre = torch.mean(pts1_pre, dim=1).squeeze(0)-R_pre @ torch.mean(pts2_pre, dim=1).squeeze(0)
            # pcd_loss = torch.mean(torch.norm(pts1_pre-pts1_gt, dim=2))#mseloss(pts1_pre, pts1_gt)#
            # x_loss = mseloss(pts1_pre[:, :, 0], pts1_gt[:, :, 0])
            # y_loss = mseloss(pts1_pre[:, :, 1], pts1_gt[:, :, 1])
            z1_loss = nn.MSELoss()(pts1_pre[:, :, 2], pts1_gt[:, :, 2])
            # z2_loss = mseloss(pts2_pre[:, :, 2], pts2_gt[:, :, 2])
            entropy_loss = (-torch.sum(weights1*torch.log(weights1+1e-10), dim=-1)).mean()#+(-torch.sum(weights2*torch.log(weights2+1e-10), dim=-1)).mean()

            t_loss = torch.linalg.norm(t.squeeze(0)-t_pre)
            R_loss = torch.abs(R.squeeze(0)-R_pre).sum()
        # print(rv12_loss, t12_loss, emd12, emd21, cyc_loss)
        # print(pcd_loss, emd12)
        # print(weights1[0, 0])
        print(z1_loss)
        pts1_gt = pts1_gt.squeeze(0).cpu().numpy()
        pts1_pre = pts1_pre.squeeze(0).cpu().numpy()
        # pcd1_c = pcd1.squeeze(0).detach().cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts1_gt[:, 0], pts1_gt[:, 1], pts1_gt[:, 2], c='g')
        # ax.scatter(pcd1_c[:10, 0], pcd1_c[:10, 1], pcd1_c[:10, 2], c='b')
        ax.scatter(pts1_pre[:, 0], pts1_pre[:, 1], pts1_pre[:, 2], c='r')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

        if i == 3:
            break