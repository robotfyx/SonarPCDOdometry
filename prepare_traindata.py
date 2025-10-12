import numpy as np
import transforms3d as t3d
import math
from utils import *
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
import collections
from tqdm import trange
import pickle
import os

N = 200 # The number of (r, theta) pairs
n_arc = 10
theta_range = math.radians(60)
phi_range = math.radians(12)
N_seeds = 80 # The number of seeds to generate voronoi diagram
keep_ratio = 0.4 # rate to keep polygons
point_noise_range = 0.01 # 1cm
rot_noise_range = math.radians(10)
trans_noise_range = 0.05 # 5cm
rng = np.random.default_rng(None)

def random_in_poly(poly: Polygon, n):
    minx, miny, maxx, maxy = poly.bounds
    pts = rng.uniform((minx, miny), (maxx, maxy), (n*3, 2))
    prep_poly = prep(poly)
    mask = np.array([prep_poly.contains(Point(x, y)) for x, y in pts])
    return pts[mask][:n]

def get_voronoi(seeds):
    # Voronoi
    vor = Voronoi(seeds)
    polygons = []
    for reg in vor.regions:
        if -1 not in reg and reg:
            verts = vor.vertices[reg]
            poly = Polygon(verts)
            if poly.is_valid and poly.area > 1e-4:
                polygons.append(poly)

    # -------------------- 3. 建立邻接图 --------------------
    # 共享 Voronoi ridge 即相邻
    ridge_dict = collections.defaultdict(set)
    for (i, j) in vor.ridge_points:
        ridge_dict[i].add(j)
        ridge_dict[j].add(i)
    # 把 polygon 索引映射回 seed 索引
    seed_centers = seeds
    poly_to_seed = [np.argmin(np.linalg.norm(seed_centers - np.array(p.centroid.xy).T[0], axis=1))
                    for p in polygons]

    adj = collections.defaultdict(set)
    for pid, sid in enumerate(poly_to_seed):
        for neighbor_sid in ridge_dict[sid]:
            try:
                nid = poly_to_seed.index(neighbor_sid)
                adj[pid].add(nid)
                adj[nid].add(pid)
            except ValueError:
                continue   # 邻居不在有效列表里

    # -------------------- 4. 相邻游走选瓦片 --------------------
    n_keep = int(len(polygons) * keep_ratio)
    start = rng.integers(len(polygons))
    visited = {start}
    queue   = collections.deque([start])

    while len(visited) < n_keep and queue:
        cur = queue.popleft()
        # 随机打乱邻居，保证游走随机
        neighbors = list(adj[cur] - visited)
        if not neighbors:
            continue
        nxt = rng.choice(neighbors)
        visited.add(nxt)
        queue.append(nxt)

    total_area = sum(polygons[i].area for i in visited)
    points = np.vstack([random_in_poly(polygons[i], max(15, int(N * polygons[i].area / total_area)))
                        for i in visited])    
    return points

def produce_data():
    # seeds in sector
    r_s = rng.uniform(0, 1, size=N_seeds)
    theta_s = rng.uniform(-theta_range, theta_range, size=N_seeds)
    seeds = np.stack((r_s*np.cos(theta_s), r_s*np.sin(theta_s)), axis=1)

    points = get_voronoi(seeds)

    rs = np.linalg.norm(points, axis=1)
    thetas = np.arctan2(points[:, 1], points[:, 0])
    r_mask = rs <= 1
    theta_mask = np.logical_and(thetas>=-theta_range, thetas<=theta_range)
    mask = np.logical_and(r_mask, theta_mask)
    # rs = np.random.uniform(0, 1, size=N).reshape((N, 1))
    # thetas = np.random.uniform(-theta_range, theta_range, size=N).reshape((N, 1))
    trueN = len(rs[mask])
    r_theta = np.stack((rs[mask], thetas[mask]), axis=1) # [trueN, 2]
    r_theta = np.expand_dims(r_theta, axis=1).repeat(n_arc, axis=1) # [trueN, n_arc, 2]
    phi = np.linspace(-phi_range, phi_range, n_arc).reshape((1, n_arc, 1))
    phi = np.repeat(phi, repeats=trueN, axis=0) # [trueN, n_arc, 1]
    r_theta_phi = np.concatenate((r_theta, phi), axis=2) # [trueN, n_arc, 3]
    r_theta_phi_c = r_theta_phi.reshape((-1, 3)) # copy

    pcd1 = po2car(r_theta_phi_c) # pcd1

    # sample on the arc
    selectidx = np.clip(np.round(rng.normal(n_arc/2, n_arc/6, trueN)), 0, n_arc-1).astype(int)
    select_pts_in1 = r_theta_phi[np.arange(trueN), selectidx] # in polar
    select_pts_in1 = po2car(select_pts_in1) # in cartesian
    select_pts_in1_copy = select_pts_in1.copy()

    # add noise
    point_noise = rng.uniform(-point_noise_range, point_noise_range, (trueN, 3))
    select_pts_in1[:, :3] = select_pts_in1[:, :3]+point_noise

    # generate T
    rot_noise = rng.uniform(-rot_noise_range, rot_noise_range, 3)
    rot_r = R.from_euler('XYZ', rot_noise, degrees=False)
    rot_R = rot_r.as_matrix()
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rot_R
    pose[:3, 3] = rng.uniform(-trans_noise_range, trans_noise_range, 3)

    # transform the selected points from 1 to 2
    select_pts_in2 = (np.linalg.inv(pose) @ select_pts_in1.T).T # [trueN, 4]

    # complete arc in 2
    rs_2 = np.linalg.norm(select_pts_in2[:, :3], axis=1)
    thetas_2 = np.arctan2(select_pts_in2[:, 1], select_pts_in2[:, 0])
    r_theta_2 = np.stack((rs_2, thetas_2), axis=1) # [trueN, 2]
    r_theta_2 = np.expand_dims(r_theta_2, axis=1).repeat(n_arc, axis=1) # [trueN, n_arc, 2]
    r_theta_phi_2 = np.concatenate((r_theta_2, phi), axis=2) # [trueN, n_arc, 3]
    r_theta_phi_2 = r_theta_phi_2.reshape((-1, 3))

    pcd2 = po2car(r_theta_phi_2)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax1.scatter(pcd1[:, 0], pcd1[:, 1], pcd1[:, 2], c='r', alpha=0.5)
    # ax1.scatter(select_pts_in1[:, 0], select_pts_in1[:, 1], select_pts_in1[:, 2], c='b')
    # ax1.scatter(select_pts_in2[:, 0], select_pts_in2[:, 1], select_pts_in2[:, 2], c='purple')
    # ax1.scatter(pcd2[:, 0], pcd2[:, 1], pcd2[:, 2], c='g', alpha=0.5)
    # # ax.scatter(pcd2[::100, 0], pcd2[::100, 1], pcd2[::100, 2], c='b')
    # ax1.set_aspect('equal')
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Y')

    # ax2 = fig.add_subplot(122, projection='polar')
    # ax2.scatter(thetas, rs, c='r')
    # ax2.set_thetamin(-60)
    # ax2.set_thetamax(60)
    # ax2.set_rmax(1)

    return pcd1, pcd2, pose, select_pts_in1_copy, select_pts_in2

if __name__ == '__main__':
    savepath = '/media/kemove/043E0D933E0D7F44/Sonar_Diffusion_SLAM/data'
    PCD1 = []
    PCD2 = []
    F1 = []
    F2 = []
    Poses = []
    Select_pts_in1 = []
    Select_pts_in2 = []
    for i in trange(10000):
        pcd1, pcd2, pose, select_pts_in1, select_pts_in2 = produce_data()
        PCD1.append(pcd1[:, :3])
        PCD2.append(pcd2[:, :3])
        n1 = estimate_normals(pcd1[:, :3])
        F1.append(n1)
        n2 = estimate_normals(pcd2[:, :3])
        F2.append(n2)
        Poses.append(pose)
        Select_pts_in1.append(select_pts_in1)
        Select_pts_in2.append(select_pts_in2)
    # PCD1 = np.array(PCD1)
    # PCD2 = np.array(PCD2)
    # Poses = np.array(Poses)
    # Select_pts_in1 = np.array(Select_pts_in1)
    data = {
        "pcd1": PCD1,
        "pcd2": PCD2,
        "feature1": F1,
        "feature2": F2,
        "poses": Poses,
        "select_pts_in1": Select_pts_in1,
        "select_pts_in2": Select_pts_in2
    }
    with open(os.path.join(savepath, 'data.pkl'), 'wb') as file:
        pickle.dump(data, file, protocol=3)