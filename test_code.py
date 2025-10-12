import torch
import torch.nn as nn
import numpy as np
from utils import rotvec2mat
from scipy.spatial.transform import Rotation as R


rot_vec = torch.tensor([[0.03, 0.068, 0.5]], requires_grad=True)  # 单个旋转

R1 = rotvec2mat(rot_vec) 
R2 = R.from_rotvec(rot_vec.detach().numpy()).as_matrix()

print(R1, '\n', R2)