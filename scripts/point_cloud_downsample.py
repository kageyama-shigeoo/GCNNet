import numpy as np
import open3d
import os
from dataset.kitti_dataset import KittiDataset
from sklearn.cluster import KMeans
from tqdm import tqdm


dataset = KittiDataset(
    '../dataset/kitti/image/training/image_2',
    '../dataset/kitti/velodyne/training/velodyne/',
    '../dataset/kitti/calib/training/calib/',
    '',
    '../dataset/kitti/3DOP_splits/val.txt',
    is_training=False)

downsample_rate = 2
output_dir = '../dataset/kitti/velodyne/training_downsampled_%d/velodyne/' % downsample_rate
for frame_idx in tqdm(range(0, dataset.num_files)):
    velo_points = dataset.get_velo_points(frame_idx)
    filename = dataset.get_filename(frame_idx)
    xyz = velo_points.xyz
    xyz_norm = np.sqrt(np.sum(xyz * xyz, axis=1, keepdims=True))
    z_axis = np.array([[0], [0], [1]])
    cos = xyz.dot(z_axis) / xyz_norm
    kmeans = KMeans(n_clusters=64, n_jobs=-1).fit(cos)
    centers = np.sort(np.squeeze(kmeans.cluster_centers_))
    centers = [-1, ] + centers.tolist() + [1, ]
    cos = np.squeeze(cos)
    point_total_mask = np.zeros(len(velo_points.xyz), dtype=np.bool)
    for i in range(0, len(centers) - 2, downsample_rate):
        lower = (centers[i] + centers[i + 1]) / 2
        higher = (centers[i + 1] + centers[i + 2]) / 2
        point_mask = (cos > lower) * (cos < higher)
        point_total_mask += point_mask
    output = np.hstack([velo_points.xyz[point_total_mask, :], velo_points.attr[point_total_mask, :]])
    point_file = output_dir + filename + '.bin'
    os.makedirs(os.path.dirname(point_file), exist_ok=True)
    output.tofile(point_file)
