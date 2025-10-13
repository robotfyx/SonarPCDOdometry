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
        shuffle=True,
        num_workers=get_dataloader_workers()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SonarPCDNet().to(device)
    checkpoints = torch.load(os.path.join(basedir, 'trainlog/exp1/model_40.pth'), map_location=device)
    model.load_state_dict(checkpoints["model"])

    for i, (pcd1, pcd2, f1, f2, rv, t, pts1_gt, pts2_gt) in enumerate(dataloader):
        with torch.no_grad():
            pcd1 = pcd1.to(device).float()
            pcd2 = pcd2.to(device).float()
            f1 = f1.to(device).float()
            f2 = f2.to(device).float()
            if pcd1.shape[1] < 2048:
                pcd1 = pad(pcd1)
                pcd2 = pad(pcd2)
                f1 = pad(f1)
                f2 = pad(f2)

            rv = rv.to(device).float()
            t = t.to(device).float()
            pts1_gt = pts1_gt.to(device).float()
            pts2_gt = pts2_gt.to(device).float()
            
            # 1->2
            rv12_pre, t12_pre, pts1_pre = model(pcd1, f1, pcd2, f2)
            r12pre = rotvec2mat(rv12_pre)

            rv12_loss = torch.norm(rv12_pre-rv, dim=1).mean()#rotation_matrix_loss(r12pre, rgt)#torch.norm(rv12_pre-rv, dim=1)
            t12_loss = torch.norm(t12_pre-t, dim=1).mean()
            # cd = chamfer_distance(pts_pre, pts_gt)
            emd12 = sinkhorn_emd(pts1_gt, pts1_pre)
            
            # 2->1
            rv21_pre, t21_pre, pts2_pre = model(pcd2, f2, pcd1, f1)
            emd21 = sinkhorn_emd(pts2_gt, pts2_pre)
            # cycle loss            
            # R12 = euler2mat(rv12_pre, deg=True)
            T12 = torch.eye(4, device=device, dtype=torch.float32)
            T12[:3, :3] = r12pre
            T12[:3, 3] = t12_pre.squeeze(0)
            # R21 = euler2mat(rv21_pre, deg=True)
            T21 = torch.eye(4, device=device, dtype=torch.float32)
            T21[:3, :3] = rotvec2mat(rv21_pre)
            T21[:3, 3] = t21_pre.squeeze(0)
            cyc_loss = torch.sum(torch.abs(T12 @ T21-torch.eye(4, device=device, dtype=torch.float32)))
        print(rv12_loss, t12_loss, emd12, emd21, cyc_loss)
        pts1_gt = pts1_gt.squeeze(0).cpu().numpy()
        pts1_pre = pts1_pre.squeeze(0).cpu().numpy()
        # pcd1_c = pcd1.squeeze(0).detach().cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts1_gt[:, 0], pts1_gt[:, 1], pts1_gt[:, 2], c='g')
        # ax.scatter(pcd1_c[:, 0], pcd1_c[:, 1], pcd1_c[:, 2], c='b')
        ax.scatter(pts1_pre[:, 0], pts1_pre[:, 1], pts1_pre[:, 2], c='r')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

        if i == 3:
            break