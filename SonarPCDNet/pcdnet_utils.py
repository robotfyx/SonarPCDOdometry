import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Any
import warnings
try:
    import SonarPCDNet._ext as _ext
except ModuleNotFoundError:
    from torch.utils.cpp_extension import load
    import glob
    import os.path as osp
    import os

    warnings.warn("Unable to load pointnet2_ops cpp extension. JIT Compiling.")

    _ext_src_root = osp.join(osp.dirname(__file__), "_ext-src")
    _ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
        osp.join(_ext_src_root, "src", "*.cu")
    )
    _ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

    os.environ["TORCH_CUDA_ARCH_LIST"] = "5.2;6.0;6.1;7.0;7.5;8.0;8.6"
    _ext = load(
        "_ext",
        sources=_ext_sources,
        extra_include_paths=[osp.join(_ext_src_root, "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
        with_cuda=True,
    )

# class SharedMLP(nn.Sequential):
#     """
#         Inherit from `nn.Sequential`\n
#         a sequence of `Conv2d`
#     """

#     def __init__(
#             self,
#             args: list[int],
#             *,
#             bn: bool = False,
#             # activation=nn.ReLU(inplace=True),
#             preact: bool = False,
#             first: bool = False,
#             name: str = "",
#             init = nn.init.kaiming_normal_ 
#     ):
#         super().__init__()

#         for i in range(len(args) - 1):
#             self.add_module(
#                 name + 'layer{}'.format(i),
#                 Conv2d(
#                     args[i],
#                     args[i + 1],
#                     bn=(not first or not preact or (i != 0)) and bn,
#                     activation=nn.LeakyReLU()
#                     if (not first or not preact or (i != 0)) else None,
#                     preact=preact,
#                     init=init
#                 )
#             )

class SharedMLP(nn.Sequential):
    def __init__(self, mlp):
        super().__init__()
        for i in range(len(mlp)-1):
            if i <= len(mlp)-3:
                self.add_module(
                    f'layer{i}',
                    nn.Conv2d(mlp[i], mlp[i+1], kernel_size=(1, 1))
                )
                self.add_module(
                    f'gn{i}',
                    nn.GroupNorm(1, mlp[i+1])
                )
                self.add_module(
                    f'act{i}',
                    nn.LeakyReLU()
                )
            elif i == len(mlp)-2:
                self.add_module(
                    f'layer{i}',
                    nn.Conv2d(mlp[i], mlp[i+1], kernel_size=(1, 1))
                )

class _ConvBase(nn.Sequential):

    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name=""
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0.0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)

class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)

class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)

class LayerNorm2d(nn.Module):
    """LayerNorm over channel dim for [B, C, N, K]"""
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x:torch.Tensor):
        # x: [B, C, N, K] -> [B, N, K, C] -> LN -> [B, C, N, K]
        y = x.permute(0, 2, 3, 1).contiguous()
        return self.ln(y).permute(0, 3, 1, 2).contiguous()

class GroupNorm2d(nn.Module):
    """GroupNorm over [B, C, N, K]"""
    def __init__(self, num_channels, num_groups=8, eps=1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels, eps=eps)
    def forward(self, x):
        # x: [B, C, N, K]
        return self.gn(x)

class Conv1d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name
        )

class Conv2d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: tuple[int, int] = (1, 1),
            stride: tuple[int, int] = (1, 1),
            padding: tuple[int, int] = (0, 0),
            activation=nn.ReLU(),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name
        )

class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        ctx.save_for_backward(idx, features)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None
gather_operation = GatherOperation.apply

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        out = _ext.furthest_point_sampling(xyz, npoint)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()
furthest_point_sample = FurthestPointSampling.apply

def _nn_distance(pc1, pc2):
    """
    Input:
        * `pc1`: (B,N,C) torch tensor
        * `pc2`: (B,M,C) torch tensor

    Output:
        * `pc_dist`: (B,N,M)
    """

    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    pc_dist = torch.sqrt(torch.sum(pc_diff**2, dim=-1) + 1e-10) # (B,N,M)

    return pc_dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_dist: grouped points distance, [B, S, nsample]
        group_idx: grouped points index, [B, S, nsample]
    """

    pc_dist = _nn_distance(new_xyz, xyz)    # [B, S, N]
    group_dist, group_idx = torch.topk(pc_dist, nsample+1, largest=False, dim=-1) # batch_size, S, nsample
    d0 = group_dist[:, :, 0]
    if torch.mean(d0) < 1e-4:
        group_dist = group_dist[:, :, 1:]
        group_idx = group_idx[:, :, 1:]
    else:
        group_dist = group_dist[:, :, :nsample]
        group_idx = group_idx[:, :, :nsample]

    group_idx = group_idx.int().contiguous()
    group_dist = group_dist.contiguous()
    
    return group_dist, group_idx

class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        ctx.save_for_backward(idx, features)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, torch.zeros_like(idx)
grouping_operation = GroupingOperation.apply

def arc_gather(pts:torch.Tensor, narc:int, nsample:int):
    B, N, _ = pts.shape
    n = int(N/narc)
    pts = pts.view(B, n, narc, 3)

    # # 为每个 (B, n) 组合生成随机不重复索引
    # perm = torch.stack([torch.randperm(narc, device=pts.device)[:nsample] for _ in range(B * n)], dim=0)
    # perm = perm.view(B, n, nsample)  # [B, n, m]
    rand_vals = torch.rand(B, n, narc, device=pts.device)
    perm = rand_vals.argsort(dim=-1)[..., :nsample]

    # 扩展索引维度方便 gather
    idx_expand = perm.unsqueeze(-1).expand(-1, -1, -1, 3)  # [B, n, m, 3]
    selected = torch.gather(pts, 2, idx_expand)
    selected = selected.view(B, n*nsample, 3)

    offset = torch.arange(n, device=pts.device).view(1, n, 1)*narc
    idx_global = (perm+offset).reshape(B, n, nsample)
    return selected, idx_global

def arc_neighbors(n:int, neighbors:int, device, B:int=1):
    base = torch.arange(n * neighbors, dtype=torch.int32).view(n, neighbors)   # [3, 4]
    idx = base.repeat_interleave(neighbors, dim=0)
    idx = idx.unsqueeze(0).tile((B, 1, 1)).to(device)
    return idx

def get_remaining(idx_global:torch.Tensor, n:int, narc:int):
    B, _, m = idx_global.shape
    device = idx_global.device
    group_offsets = torch.arange(n * narc, device=device).view(1, n, narc).expand(B, -1, -1)
    local_idx = idx_global % narc  # [B, n, m]
    local_idx_expanded = local_idx.unsqueeze(-1).expand(-1, -1, -1, narc)  # [B, n, m, narc]
    group_full_expanded = group_offsets.unsqueeze(2).expand(-1, -1, m, -1) # [B, n, m, narc]
    mask = (torch.arange(narc, device=device).view(1, 1, 1, narc) == local_idx_expanded)
    mask = mask.to(device)
    rest = group_full_expanded[~mask].view(B, n, m, narc - 1)  # [B, n, m, narc-1]
    rest = rest.view(B, n*m, narc-1)
    return rest.to(torch.int32)