import torch
import torch.nn as nn
from .pcdnet_utils import SharedMLP, gather_operation, furthest_point_sample, knn_point, grouping_operation, Conv1d
from typing import Optional
from torch.nn import functional as F
from .pose import Pose
import math

class SAModule(nn.Module):
    def __init__(self, mlp:list[int], npoint:int, nsample:int, bn:bool=True):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample

        mlp_spec = mlp
        if mlp[0] == 0:
            mlp_spec[0] += 3
        mlp_spec[0] += 3
        self.mlp_module = SharedMLP(mlp_spec, bn=bn, init=nn.init.xavier_uniform_)
    
    def forward(self, xyz:torch.Tensor, features:Optional[torch.Tensor]) ->tuple[torch.Tensor, torch.Tensor]:
        '''
        input:
        xyz: [B, N, 3] 3D coordinates
        features: [B, C, N]

        output:
        new_xyz: [B, npoint, 3] selected points
        new_features [B, mlps[-1], npoint]
        '''
        xyz_flipped = xyz.transpose(1, 2).contiguous() # [B, 3, N]
        new_xyz = (gather_operation(xyz_flipped, furthest_point_sample(xyz, self.npoint)).transpose(1, 2).contiguous()) # [B, npoint, 3]

        if features is not None:
            _, idxq = knn_point(self.nsample, xyz, new_xyz) # [B, npoint, nsample]
            grouped_xyz = grouping_operation(xyz_flipped, idxq) # [B, 3, npoint, nsample]
            grouped_features = grouping_operation(features, idxq) # [B, C, npoint, nsample]

            new_xyz_t = torch.permute(new_xyz, (0, 2, 1)).contiguous() # [B, 3, npoint]
            new_xyz_expand = torch.tile(new_xyz_t.unsqueeze(-1), (1, 1, 1, self.nsample)) # [B, 3, npoint, nsample]
            xyz_diff = grouped_xyz-new_xyz_expand
            new_features = torch.cat((xyz_diff, grouped_features), dim=1) # [B, C+3, npoint, nsample]
        else:
            _, idxq = knn_point(self.nsample, xyz, new_xyz) # [B, npoint, nsample]
            grouped_xyz = grouping_operation(xyz_flipped, idxq) # [B, 3, npoint, nsample]
            new_xyz_t = torch.permute(new_xyz, (0, 2, 1)).contiguous() # [B, 3, npoint]
            new_xyz_expand = torch.tile(new_xyz_t.unsqueeze(-1), (1, 1, 1, self.nsample)) # [B, 3, npoint, nsample]
            xyz_diff = grouped_xyz-new_xyz_expand
            new_features = torch.cat((xyz_diff, grouped_xyz), dim=1) # [B, 3+3, npoint, nsample]
        
        new_features = self.mlp_module(new_features) # [B, mlp[-1], npoint, nsmaple]
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]) # [B, mlp[-1], npoint, 1]
        new_features = new_features.squeeze(-1) # [B, mlp[-1], npoint]

        return new_xyz, new_features
    
class CostVolume(nn.Module):
    def __init__(self, nsample, nsample_q, in_channel1, in_channel2, mlp1, mlp2): # mlp[-1]=mlp1[-1]=mlp2[-1]
        super(CostVolume, self).__init__()
        self.nsample = nsample
        self.nsample_q = nsample_q
        # self.in_channel = [in_channel1, in_channel2, 10]

        mlp1_spec = [in_channel1+in_channel2+10]+mlp1
        self.mlp_convs = SharedMLP(mlp1_spec, bn=False, init=nn.init.xavier_uniform_)
        mlp_spec_xyz_1 = [10, mlp1[-1]]
        self.mlp_conv_xyz_1 = SharedMLP(mlp_spec_xyz_1, bn=False, init=nn.init.xavier_uniform_)
        mlp_spec_xyz_2 = [10, mlp2[-1]]
        self.mlp_conv_xyz_2 = SharedMLP(mlp_spec_xyz_2, bn=False, init=nn.init.xavier_uniform_)

        # concatenating 3D Euclidean space encoding and first flow embeddings
        last_channel2 = mlp1_spec[-1] * 2 
        mlp2_spec = [last_channel2] + mlp2
        self.mlp2_convs = SharedMLP(mlp2_spec, bn=False, init=torch.nn.init.xavier_uniform_)
        last_channel3 = mlp1_spec[-1] * 2 + in_channel1   
        mlp3_spec = [last_channel3] + mlp2
        self.mlp3_convs = SharedMLP(mlp3_spec, bn=False, init=torch.nn.init.xavier_uniform_)
        self.out_channel = mlp3_spec[-1]
    
    def forward(self, xyz1:torch.Tensor, feature1:torch.Tensor, xyz2:torch.Tensor, feature2:torch.Tensor):
        '''
        input:
        xyz1: [B, 3, S] points in frame1
        feature1: [B, C, S] features of the points in frame1
        xyz2: [B, 3, N] points in frame2
        feature2: [B, C, N] features of the points in frame2

        return:
        pc_feat1_new: [B, mlp[-1], S] the cost volume
        '''
        xyz1_t = xyz1.permute(0, 2, 1).contiguous() # [B, S, 3]
        xyz2_t = xyz2.permute(0, 2, 1).contiguous() # [B, N, 3]
        
        ### -----------------------------------------------------------
        ### FIRST AGGREGATE
        _, idx_q = knn_point(self.nsample_q, xyz2_t, xyz1_t) # [B, S, k] k=nsample_q

        # -- ME --
        qi_xyz_grouped = grouping_operation(xyz2, idx_q) # [B, 3, S, k]
        qi_points_grouped = grouping_operation(feature2, idx_q) # [B, C2, S, k]

        pi_xyz_expanded = torch.tile(torch.unsqueeze(xyz1, 3), [1, 1, 1, self.nsample_q]) # [B, 3, S, k]
        pi_points_expanded = torch.tile(torch.unsqueeze(feature1, 3), [1, 1, 1, self.nsample_q]) # [B, C1, S, k]
        pi_xyz_diff = qi_xyz_grouped - pi_xyz_expanded # [B, 3, S, k]        
        pi_euc_diff = torch.sqrt(torch.sum(torch.square(pi_xyz_diff), dim=1, keepdim=True)+1e-20) # [B, 1, S, k]
    
        pi_xyz_diff_concat = torch.cat((pi_xyz_expanded, qi_xyz_grouped, pi_xyz_diff, pi_euc_diff), dim=1) # [B, 3+3+3+1, S, k] = [B, 10, S, k]
        
        pi_feat_diff = torch.cat((pi_points_expanded, qi_points_grouped), dim=1) # [B, C1 + C2, S, k]
        pi_feat1_new = torch.cat((pi_xyz_diff_concat, pi_feat_diff), dim=1) # [B, 10 + C1 + C2, S, k]

        # pi_feat1_new
        # the first flow embedding uses 3D Euclidean space information `pi_xyz_diff_concat` and
        # the features from the two frames of points clouds: `pi_points_expanded` and `qi_points_grouped`
        pi_feat1_new = self.mlp_convs(pi_feat1_new) # [B, mlp1[-1], S, k]

        # The spatial structure information `pi_xyz_diff_concat` not only helps
        # to determine the similarity of points, but also can contribute to deciding 
        # soft aggregation weights of the queried points

        pi_xyz_encoding = self.mlp_conv_xyz_1(pi_xyz_diff_concat) # [B, mlp1[-1], S, k]
        pi_concat = torch.cat((pi_xyz_encoding, pi_feat1_new), dim = 1) # [B, 2 * mlp1[-1], S, k]
        pi_concat = self.mlp2_convs(pi_concat) # [B, mlp2[-1], S, k]
        WQ = F.softmax(pi_concat, dim=3) # [B, mlp2[-1], S, k]
            
        # `pi_feat1_new` are The first attentive flow embeddings
        pi_feat1_new = WQ * pi_feat1_new
        pi_feat1_new = torch.sum(pi_feat1_new, dim=3, keepdim=False) # [B, mlp[-1], S]

        ### -----------------------------------------------------------
        ### SECOND AGGREGATE
        _, idx = knn_point(self.nsample, xyz1_t, xyz1_t) # [B, S, m] m=nsample
        pc_xyz_grouped = grouping_operation(xyz1, idx) # [B, 3, S, m]
        pc_points_grouped = grouping_operation(pi_feat1_new, idx) # [B, mlp[-1], S, m]

        pc_xyz_new = torch.tile( torch.unsqueeze(xyz1, 3), [1, 1, 1, self.nsample]) # [B, 3, S, m]
        pc_points_new = torch.tile( torch.unsqueeze(feature1, 3), [1, 1, 1, self.nsample]) # [B, C1, S, m]

        pc_xyz_diff = pc_xyz_grouped-pc_xyz_new # [B, 3, S, m]
        pc_euc_diff = torch.sqrt(torch.sum(torch.square(pc_xyz_diff), dim=1, keepdim=True) + 1e-20) # [B, 1, S, m]
        pc_xyz_diff_concat = torch.cat((pc_xyz_new, pc_xyz_grouped, pc_xyz_diff, pc_euc_diff), dim=1) # [B, 10, S, m]

        pc_xyz_encoding = self.mlp_conv_xyz_2(pc_xyz_diff_concat) # [B, mlp2[-1], S, m]

        pc_concat = torch.cat((pc_xyz_encoding, pc_points_new, pc_points_grouped), dim = 1) # [B, mlp1[-1] + C1 + mlp2[-1], S, m]
        pc_concat = self.mlp3_convs(pc_concat) # [B, mlp2[-1], S, m]
        WP = F.softmax(pc_concat, dim=3) # [B, mlp2[-1], S, m]

        # The final attentive flow embedding
        # pc_feat1_new: (B, mlp[-1], S)
        # mlp[-1] = mlp1[-1] = mlp2[-1]

        pc_feat1_new = WP * pc_points_grouped
        pc_feat1_new = torch.sum(pc_feat1_new, dim=3, keepdim=False) # [B, mlp[-1], S]

        return pc_feat1_new

class OPPredictor(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        # self.in_channel = [in_channel]
        mlp_spec = [in_channel]+mlp
        self.mlp_convs = SharedMLP(mlp_spec, bn=False, init=nn.init.xavier_uniform_)
        # self.out_channel = mlp_spec[-1]

    def forward(self, feature1:torch.Tensor, cost_volume:torch.Tensor, upsampled_feat=None):
        if feature1 is None:
            points_concat = cost_volume
        elif upsampled_feat != None:
            points_concat = torch.cat((feature1, cost_volume, upsampled_feat), dim=1) # [B, C1 + C2 + C', N]
        elif upsampled_feat == None:
            points_concat = torch.cat((feature1, cost_volume), dim=1) # [B, C1 + C2, N]
        points_concat = torch.unsqueeze(points_concat, 3) # [B, C1 + C2 + (C'|0), N, 1]
        points_concat = self.mlp_convs(points_concat) # [B, mlp[-1], N, 1]                                        
        points_concat = torch.squeeze(points_concat, dim=3) # [B, mlp[-1], N]

        # OP = F.softmax(points_concat, dim=1) # [B, mlp[-1], N]
        OP = F.sigmoid(points_concat)
        return OP #points_concat
    
class PosePredictor(nn.Module):
    def __init__(self, in_channel:int, out_channel:int):
        super().__init__()
        # self.pose = pose
        # self.squeeze = squeeze
        # self.conv1d_q_t = Conv1d(in_size=in_channel, out_size=out_channel, kernel_size=kernel_size, padding=padding, activation=activation, init=torch.nn.init.xavier_uniform_)
        # self.conv1d_q = Conv1d(in_size=out_channel, out_size=4, kernel_size=kernel_size, padding=padding, activation=activation, init=torch.nn.init.xavier_uniform_)    # 4 instead of self.pose.num_rot_params()
        # self.conv1d_t = Conv1d(in_size=out_channel, out_size=3, kernel_size=kernel_size, padding=padding, activation=activation, init=torch.nn.init.xavier_uniform_)
        self.cost_volume_encoder = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            # nn.Linear(in_features=256, out_features=256),
            # nn.ReLU(),
            nn.Linear(in_features=256, out_features=out_channel)
        )
        self.rv_decoder = nn.Sequential(
            nn.Linear(in_features=out_channel, out_features=256),
            nn.ReLU(),
            # nn.Linear(in_features=256, out_features=256),
            # nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=3)
        )
        self.t_decoder = nn.Sequential(
            nn.Linear(in_features=out_channel, out_features=256),
            nn.ReLU(),
            # nn.Linear(in_features=256, out_features=256),
            # nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=3)
        )
        self.pcd_decoder1 = nn.Sequential(
            nn.Conv2d(out_channel+2, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=1),
        )
        self.pcd_decoder2 = nn.Sequential(
            nn.Conv2d(out_channel+3, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=1),
        )


    def forward(self, embedding_features, mask, numpoints):
        """
        Inputs:
            * embedding_features:   [B, C, N]
            * mask:                 [B, C, N]
        Return:
            * q:                    [B, 4] if squeeze else [B, 4, 1]
            * t:                    [B, 3] if squeeze else [B, 3, 1]
        """
        cost_volume_sum = torch.sum(embedding_features * mask, dim=2, keepdim=True) # [B, C, 1]
        # cost_volume_sum_big = self.conv1d_eu_t(cost_volume_sum) # [B, self.conv1d_q_t.out_channel, 1]
        cost_volume_sum_big = self.cost_volume_encoder(cost_volume_sum.squeeze(2)) #[B, Cout]

        # # cost_volume_sum_q: [B, self.conv1d_q_t.out_channel, 1]
        # cost_volume_sum_q = F.dropout(cost_volume_sum_big, p=0.5, training=self.training)
        # # cost_volume_sum_t: [B, self.conv1d_q_t.out_channel, 1]
        # cost_volume_sum_t = F.dropout(cost_volume_sum_big, p=0.5, training=self.training)

        # q: [B, 4, 1]
        # q = self.conv1d_q(cost_volume_sum_q)
        # eu = self.conv1d_eu(cost_volume_sum_big)
        rv = self.rv_decoder(cost_volume_sum_big) # [B, 3]
        # q = q / (torch.sqrt(torch.sum(q*q, dim=1, keepdim=True) + 1e-10) + 1e-10)
        
        # t: [B, 3, 1]
        # t = self.conv1d_t(cost_volume_sum_t)
        # t = self.conv1d_t(cost_volume_sum_big)
        t = self.t_decoder(cost_volume_sum_big) # [B, 3]

        # assert len(q.size()) == len(t.size()) == 3, f'[Pose Calculator] Wrong shape q={q.size()} and t={t.size()}'
        # assert q.size(0) == t.size(0) == embedding_features.size(0), f'[Pose Calculator] Wrong shape q={q.size()} and t={t.size()}'
        # assert q.size(1) == 4, f'[Pose Calculator] Wrong shape q: {q.size()}'
        # assert t.size(1) == 3, f'[Pose Calculator] Wrong shape t: {t.size()}'
        # assert q.size(2) == t.size(2) == 1, f'[Pose Calculator] Wrong shape q={q.size()} and t={t.size()}'

        # if self.squeeze:
        #     # q: [B, 4]
        #     eu = torch.squeeze(eu, dim=2)
        #     # t: [B, 3]
        #     t = torch.squeeze(t, dim=2)
        
        # pcd decoder
        grid_size = int(math.sqrt(numpoints))
        embedding = torch.unsqueeze(torch.unsqueeze(cost_volume_sum_big, dim=2), dim=3) # [B, self.conv1d_q_t.out_channel, 1, 1]
        embedding = torch.tile(embedding, [1, 1, grid_size, grid_size]) # [B, self.conv1d_q_t.out_channel, grid_size, grid_size]
        B = embedding.shape[0]
        # 2d grid
        u = torch.linspace(-1, 1, grid_size)
        v = torch.linspace(-1, 1, grid_size)
        uu, vv = torch.meshgrid(u, v, indexing="ij")
        grid = torch.stack([uu, vv], dim=-1) # [grid_size, grid_size, 2]
        grid = torch.tile(grid.unsqueeze(0), [B, 1, 1, 1]).permute(0, 3, 1, 2) # [B, 2, grid_size, grid_size]
        feat1 = torch.cat((embedding, grid.to(embedding.device)), dim=1)

        folding1 = self.pcd_decoder1(feat1) # [B, 3, grid_size, grid_size]
        feat2 = torch.cat((embedding, folding1), dim=1) # [B, C+3, ..., ...]
        points_out = self.pcd_decoder2(feat2) # [B, 3, grid_size, grid_size]
        points_out = points_out.permute(0, 2, 3, 1).reshape(-1, grid_size*grid_size, 3)
        
        return rv, t, points_out
    