import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import os
from transforms3d import quaternions, euler
from scipy.spatial.transform import Rotation as R

class SonarPCDData(Dataset):
    def __init__(self, datadir):
        with open(os.path.join(datadir, 'data.pkl'), 'rb') as file:
            self.data = pickle.load(file)
        self.pcd1 = self.data["pcd1"]
        self.pcd2 = self.data["pcd2"]
        self.f1 = self.data["feature1"]
        self.f2 = self.data["feature2"]
        self.posegt = self.data["poses"]
        self.select_pts_in1 = self.data["select_pts_in1"]
        self.select_pts_in2 = self.data["select_pts_in2"]
    def __len__(self):
        return len(self.posegt)
    def __getitem__(self, index):
        pcd1 = self.pcd1[index]
        pcd2 = self.pcd2[index]
        f1 = self.f1[index]
        f2 = self.f2[index]
        pose = self.posegt[index]
        # q = quaternions.mat2quat(pose[:3, :3])
        # eu = R.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True)
        rv = R.from_matrix(pose[:3, :3]).as_rotvec(degrees=False)
        t = pose[:3, 3]

        pts1 = self.select_pts_in1[index][:, :3]
        pts2 = self.select_pts_in2[index][:, :3]

        return torch.from_numpy(pcd1), torch.from_numpy(pcd2), torch.from_numpy(f1), torch.from_numpy(f2), torch.from_numpy(rv), torch.from_numpy(t), torch.from_numpy(pts1), torch.from_numpy(pts2)