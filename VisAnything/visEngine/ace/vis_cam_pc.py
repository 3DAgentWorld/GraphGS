import os
from ace_visualizer import ACEVisualizer
from plyfile import PlyData, PlyElement
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def vis_cam_train(save_path, pose_rgb_file, ply_file, all_iter):
    ace_visualizer = ACEVisualizer(
        save_path,
        False,
        10,
        mapping_vis_error_threshold=10)

    ace_visualizer.setup_mapping_visualisation_direct(
        pose_rgb_file,
        all_iter // 100 + 1,
        4  # camera_z_offset
    )

    pcd = fetchPly(ply_file)

    for i in range(all_iter):
        if i % 100 == 0:
            ace_visualizer.render_mapping_frame_direct(pcd)

    # ace_visualizer.finalize_mapping(self.regressor, vis_dataset_loader)


if __name__ == "__main__":
    save_path = 'visEngine/ace/ace_output'
    pose_rgb_file = 'colmap_cam_demo/waymo_10061/exhaustive/camera_info_opencv.json'
    ply_file = 'colmap_cam_demo/waymo_10061/exhaustive/sparse/0/points3D.ply'
    all_iter = 1000
    # pcd = fetchPly(ply_file)
    # import ipdb
    # ipdb.set_trace()
    os.makedirs(save_path, exist_ok=True)
    vis_cam_train(save_path, pose_rgb_file, ply_file, all_iter)
