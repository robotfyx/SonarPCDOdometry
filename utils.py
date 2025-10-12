import numpy as np
import torch
import torch.nn as nn
from typeguard import check_type
from typing import Optional, Union
TensorType = Union[torch.Tensor, np.ndarray]
import sys
from scipy.spatial import cKDTree
import math

def po2car(pts):
    '''
    input: points in polar [N, 3] r theta phi
    output: points in cartesian [N, 3] x y z
    '''
    x = pts[:, 0]*np.cos(pts[:, 1])*np.cos(pts[:, 2])
    y = pts[:, 0]*np.sin(pts[:, 1])*np.cos(pts[:, 2])
    z = pts[:, 0]*np.sin(pts[:, 2])
    i = np.ones(len(pts))
    return np.stack((x, y, z, i), axis=-1)

def assert_debug(condition: bool, message: str = ""):
    """
    Debug Friendly assertion

    Allows to put a breakpoint, and catch any assertion error in debug
    """
    if not condition:
        print(f"[ERROR][ASSERTION]{message}")
        raise AssertionError(message)
    
def sizes_match(tensor, sizes: list) -> bool:
    """
    Returns True if the sizes matches the tensor shape
    """
    tensor_shape = list(tensor.shape)
    if len(tensor_shape) != len(sizes):
        return False
    for i in range(len(sizes)):
        if sizes[i] != -1 and sizes[i] != tensor_shape[i]:
            return False
    return True

def check_tensor(tensor: tuple[torch.Tensor, np.ndarray], sizes: list, tensor_type: Optional[type] = TensorType):
    """
    Checks the size of a tensor along all its dimensions, against a list of sizes

    The tensor must have the same number of dimensions as the list sizes
    For each dimension, the tensor must have the same size as the corresponding entry in the list
    A size of -1 in the list matches all sizes

    Optionally it checks the type of the tensor (either np.ndarray or torch.Tensor)

    Any Failure raises an AssertionError

    >>> check_tensor(torch.randn(10, 3, 4), [10, 3, 4])
    >>> check_tensor(torch.randn(10, 3, 4), [-1, 3, 4])
    >>> check_tensor(np.random.randn(2, 3, 4), [2, 3, 4])
    >>> #torch__check_sizes(torch.randn(10, 3, 4), [9, 3, 4]) # --> throws an AssertionError
    """
    assert_debug(sizes_match(tensor, sizes),
                 f"[BAD TENSOR SHAPE] Wrong tensor shape got {tensor.shape} expected {sizes}")
    if tensor_type is not None:
        # -- ME -- commented next line
        #check_type("tensor", tensor, tensor_type)
        check_type(tensor, tensor_type)

def get_dataloader_workers():
    return 0 if sys.platform.startswith('win') else 4

def estimate_normals(points, k=16, orient=True):
    """
    估计点云的法向量 (基于 PCA)

    参数:
        points: (N, 3) numpy数组，点云
        k: 邻域点数目 (默认16)
        orient: 是否进行法向量方向统一 (默认True)

    返回:
        normals: (N, 3) numpy数组，单位化后的法向量
    """
    N = points.shape[0]
    tree = cKDTree(points)
    normals = np.zeros((N, 3))

    for i in range(N):
        # 查询邻域点索引
        _, idx = tree.query(points[i], k=k)
        neighbors = points[idx]

        # 去中心化
        neighbors_centered = neighbors - neighbors.mean(axis=0)

        # 协方差矩阵
        cov = np.dot(neighbors_centered.T, neighbors_centered)

        # 特征值分解
        eigvals, eigvecs = np.linalg.eigh(cov)

        # 取最小特征值对应的特征向量
        normal = eigvecs[:, np.argmin(eigvals)]
        normal = normal / np.linalg.norm(normal)

        normals[i] = normal

    if orient:
        # 法向量方向统一: 朝外（远离点云质心）
        center = np.mean(points, axis=0)
        for i in range(N):
            vec = points[i] - center
            if np.dot(normals[i], vec) < 0:
                normals[i] = -normals[i]

    return normals

def pad(pts:torch.Tensor)->torch.Tensor:
    '''
    pts shape [B, N, 3]
    '''
    B, N, _ = pts.shape
    need = 2048-N
    last = pts[:, -1:, :]
    tail = last.repeat(1, need, 1)
    return torch.cat([pts, tail], dim=1)

def chamfer_distance(pcl1, pcl2):
    """
    计算两个点云的Chamfer Distance（支持梯度反传）
    输入:
        pcl1: [B, N1, 3]
        pcl2: [B, N2, 3]
    输出:
        dist: 标量 (batch mean)
    """
    B, N1, _ = pcl1.shape
    B, N2, _ = pcl2.shape

    # 计算 pairwise L2 距离矩阵: [B, N1, N2]
    diff = pcl1.unsqueeze(2) - pcl2.unsqueeze(1)  # [B, N1, N2, 3]
    dist_matrix = torch.sum(diff ** 2, dim=-1)    # [B, N1, N2]

    # 每个点找到最近邻
    min_dist1, _ = torch.min(dist_matrix, dim=2)  # [B, N1]
    min_dist2, _ = torch.min(dist_matrix, dim=1)  # [B, N2]

    # 平均 Chamfer 距离
    cd = (min_dist1.mean(dim=1) + min_dist2.mean(dim=1)).mean()
    return cd

def sinkhorn_emd(P, Q, eps=0.01, n_iters=50):
    """
    Differentiable approximate EMD via Sinkhorn algorithm.
    Args:
        P: [B, N1, 3] point cloud 1
        Q: [B, N2, 3] point cloud 2
        eps: entropy regularization coefficient
        n_iters: number of Sinkhorn iterations
    Returns:
        emd_loss: [B] differentiable scalar EMD (per batch)
    """
    B, N1, _ = P.shape
    _, N2, _ = Q.shape

    # cost matrix: pairwise L2 distance
    C = torch.cdist(P, Q, p=2)  # [B, N1, N2]

    # uniform distributions over points
    mu = torch.full((B, N1), 1.0 / N1, device=P.device, dtype=P.dtype)
    nu = torch.full((B, N2), 1.0 / N2, device=P.device, dtype=P.dtype)

    # log-domain Sinkhorn iterations (stabilized)
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)

    for _ in range(n_iters):
        u = eps * (torch.log(mu + 1e-8) - torch.logsumexp(M(C, u, v, eps), dim=2)) + u
        v = eps * (torch.log(nu + 1e-8) - torch.logsumexp(M(C, u, v, eps).transpose(1,2), dim=2)) + v

    # transport plan
    pi = torch.exp(M(C, u, v, eps))  # [B, N1, N2]
    # Sinkhorn distance
    emd = torch.sum(pi * C, dim=[1,2])
    return emd

def M(C, u, v, eps):
    # helper for stabilized computation
    return (-C + u.unsqueeze(2) + v.unsqueeze(1)) / eps

def quat2mat(q: torch.Tensor)->torch.Tensor:
    '''
    input: q [B, 4]
    out: R [B, 3, 3]
    '''
    # 展开分量
    w, x, y, z = q.unbind(dim=-1)      # 每个 (...,)

    # 预计算重复出现的乘积
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz = x * y, x * z
    yz = y * z

    # 构造旋转矩阵
    rot = torch.stack([
        torch.stack([ww + xx - yy - zz, 2 * (xy - wz),     2 * (xz + wy)],    dim=-1),
        torch.stack([2 * (xy + wz),     ww - xx + yy - zz, 2 * (yz - wx)],    dim=-1),
        torch.stack([2 * (xz - wy),     2 * (yz + wx),     ww - xx - yy + zz], dim=-1),
    ], dim=-2)  # (..., 3, 3)

    return rot

def euler2mat(angle: torch.Tensor, deg:bool=False) -> torch.Tensor:
    """
    将批量欧拉角 (deg) 转换为批量 3x3 旋转矩阵（支持梯度）。
    
    参数
    ----
    b_euler_deg : torch.Tensor, shape [B, 3]
        每行是 (rx_deg, ry_deg, rz_deg)，表示绕 x, y, z 轴的旋转角（度）。
        语义（默认约定）：
          - 先绕 x 轴旋转 rx（Rx）
          - 再绕 y 轴旋转 ry（Ry）
          - 最后绕 z 轴旋转 rz（Rz）
        对列向量 v 的作用顺序为： v' = Rz @ Ry @ Rx @ v
    
    返回
    ----
    torch.Tensor, shape [B, 3, 3]
        对应的批量旋转矩阵（float32/float64 同输入）。
    """
    # 保持 dtype / device
    dtype = angle.dtype
    device = angle.device
    if deg:
        # deg -> rad (可微)
        rad = angle * (math.pi / 180.0)
    else:
        rad = angle

    rx = rad[:, 0]
    ry = rad[:, 1]
    rz = rad[:, 2]

    cx = torch.cos(rx); sx = torch.sin(rx)
    cy = torch.cos(ry); sy = torch.sin(ry)
    cz = torch.cos(rz); sz = torch.sin(rz)

    B = angle.shape[0]

    # 构造 Rx, Ry, Rz（batch-wise）
    # Rx = [[1, 0, 0],
    #       [0, cx, -sx],
    #       [0, sx,  cx]]
    Rx = torch.zeros((B, 3, 3), dtype=dtype, device=device)
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1] = cx
    Rx[:, 1, 2] = -sx
    Rx[:, 2, 1] = sx
    Rx[:, 2, 2] = cx

    # Ry = [[ cy, 0, sy],
    #       [  0, 1,  0],
    #       [-sy, 0, cy]]
    Ry = torch.zeros((B, 3, 3), dtype=dtype, device=device)
    Ry[:, 0, 0] = cy
    Ry[:, 0, 2] = sy
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = -sy
    Ry[:, 2, 2] = cy

    # Rz = [[cz, -sz, 0],
    #       [sz,  cz, 0],
    #       [ 0,   0, 1]]
    Rz = torch.zeros((B, 3, 3), dtype=dtype, device=device)
    Rz[:, 0, 0] = cz
    Rz[:, 0, 1] = -sz
    Rz[:, 1, 0] = sz
    Rz[:, 1, 1] = cz
    Rz[:, 2, 2] = 1.0

    # 合成 R = Rz @ Ry @ Rx
    Rzy = torch.bmm(Rz, Ry)   # [B,3,3]
    R = torch.bmm(Rzy, Rx)    # [B,3,3]

    return R

def rotvec2mat(rot_vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert a batch of rotation vectors (axis-angle) to rotation matrices (B x 3 x 3).
    Differentiable implementation (Rodrigues' formula).

    Args:
        rot_vec: Tensor of shape [B, 3], rotation vectors (axis * angle)
        eps: Small constant to avoid division by zero

    Returns:
        rot_mat: Tensor of shape [B, 3, 3], rotation matrices
    """
    # rotation angle (B, 1)
    theta = torch.norm(rot_vec, dim=1, keepdim=True).clamp(min=eps)

    # normalized rotation axis (B, 3)
    k = rot_vec / theta

    # components
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]

    # skew-symmetric matrix [B, 3, 3]
    zero = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([zero, -kz, ky], dim=-1),
        torch.stack([kz, zero, -kx], dim=-1),
        torch.stack([-ky, kx, zero], dim=-1)
    ], dim=1)  # (B, 3, 3)

    # Rodrigues' rotation formula
    I = torch.eye(3, device=rot_vec.device).unsqueeze(0)  # (1, 3, 3)
    sin_theta = torch.sin(theta).unsqueeze(-1)  # (B, 1, 1)
    cos_theta = torch.cos(theta).unsqueeze(-1)

    R = I + sin_theta * K + (1 - cos_theta) * (K @ K)
    return R

def rotation_matrix_loss(R_pred:torch.Tensor, R_gt:torch.Tensor):
    cos_theta = ((R_pred.transpose(1,2) @ R_gt).diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2
    cos_theta = cos_theta.clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos_theta).mean()