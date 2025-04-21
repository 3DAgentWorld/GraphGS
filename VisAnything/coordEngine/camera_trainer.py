import os

from kornia.geometry.depth import depth_to_3d, depth_to_normals, depth_to_3d_v2
from coordEngine.colmap_utils import BasicPointCloud, save_basicpcd, increase_pcd
from PIL import Image
import numpy as np
import cv2
from collections import defaultdict, OrderedDict
import torch
import open3d as o3d

torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)


class DepthPredictor(object):

    def __init__(self, cam_list_obj, depth_model_type='dpt'):
        self.depth_model_type = depth_model_type  # better: dpt
        self.near = 0.01

        self.setup_depth_predictor()
        self.mono_depth = OrderedDict()
        self.mono_pcd = OrderedDict()
        self.cam_list_obj = cam_list_obj

    def generate_one_pcd(self, idx, data_dir=None, down_sample=False):
        cam0 = self.cam_list_obj.get_camera_list()[idx]
        image_np = self.read_img_from_path(cam0.image_path)
        depth_tensor = self.predict_depth_from_np(image_np, idx)
        pcd = self.depth_K_to_pcd(depth_tensor, cam0.K, image_np, idx, cam0.P_c2w, down_sample)
        print(f'pcd {idx:04d}/{len(self.cam_list_obj):04d} done, shape: {pcd.points.shape}')
        if data_dir is not None:
            save_dir = os.path.join(data_dir, 'mono_depth_pcd')
            save_path = os.path.join(save_dir, f'idx{idx:04d}_{cam0.image_name}.ply')
            os.makedirs(save_dir, exist_ok=True)
            save_basicpcd(pcd, save_path)
            print(f'saved .ply to {save_path}')
        return pcd

    def generate_multi_pcd(self, frames, data_dir=None):
        self.frames = frames
        frame_start = frames[0]
        frame_end = frames[1]
        frame_gap = frames[2]
        idx_list = [i for i in range(len(self.cam_list_obj))]
        idx_list = idx_list[frame_start:frame_end:frame_gap]
        pcd_start = None
        for idx in idx_list:
            next_pcd = self.generate_one_pcd(idx, data_dir=None, down_sample=True)
            if pcd_start is None:
                pcd_start = next_pcd
                continue
            pcd_start = increase_pcd(pcd_start, next_pcd)

        self.merged_pcd = pcd_start
        if data_dir is not None:
            save_dir = os.path.join(data_dir, 'mono_depth_pcd')
            save_path = os.path.join(save_dir, f'merge_{frame_start:04d}-{frame_end:04d}-{frame_gap:04d}.ply')
            os.makedirs(save_dir, exist_ok=True)
            save_basicpcd(pcd_start, save_path, to_blender=True)
            print(f'saved .ply to {save_path}')
            self.cam_list_obj.save_frame_to_json(save_dir, frames)
        return pcd_start

    def setup_depth_predictor(self):
        # we recommand to use the following depth models:
        # - "midas" for the Tank and Temples dataset
        # - "zoe" for the CO3D dataset
        # - "depth_anything" for the custom dataset
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.depth_model_type == "zoe":
            repo = "isl-org/ZoeDepth"
            model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
            zoe = model_zoe_n.to(device)
            self.depth_model = zoe
        elif self.depth_model_type == "depth_anything":
            # maybe install depth anything
            from torchvision.transforms import Compose
            from submodules.DepthAnything.depth_anything.dpt import DepthAnything
            from submodules.DepthAnything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
            encoder = 'vits'  # can also be 'vitb' or 'vitl'
            depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()
            # depth_anything = DepthAnything.from_pretrained('checkpoints/depth_anything_metric_depth_outdoor', local_files_only=True).eval()

            self.depth_transforms = Compose([
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
            self.depth_model = depth_anything
        else:
            midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            midas.to(device)
            midas.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.depth_transforms = midas_transforms.dpt_transform
            self.depth_model = midas

    def read_img_from_path(self, img_path):
        image_pil = Image.open(img_path).convert("RGB")  # (0,255)
        image_np = np.asarray(image_pil) / 255.0  # (0,1) H W C
        image_torch = torch.from_numpy(
            image_np).permute(2, 0, 1).float()  # (0,1) C H W
        return image_np

    def predict_depth_from_np(self, image_np, idx):
        if idx not in self.mono_depth:
            depth_tensor = self.predict_depth(np.asarray(image_np) * 255)
            # depth_tensor = self.depth_model.infer_pil(image_pil, output_type='tensor')
            depth_tensor[depth_tensor < self.near] = self.near
            self.mono_depth[idx] = depth_tensor.cuda()
        else:
            depth_tensor = self.mono_depth[idx]
        return depth_tensor  # (h,w)

    def predict_depth(self, img):
        if self.depth_model_type == "zoe":
            depth = self.depth_model.infer_pil(Image.fromarray(img.astype(np.uint8)),
                                               output_type='tensor')
        elif self.depth_model_type == "depth_anything":
            image = self.depth_transforms({'image': img / 255.})['image']
            image = torch.from_numpy(image).unsqueeze(0)
            # depth shape: 1xHxW
            prediction = self.depth_model(image)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().detach()
            # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

            # depth = depth.cpu().numpy().astype(np.uint8)
            # depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            # pdb.set_trace()
            scale = 0.0305
            shift = 0.15
            depth = scale * prediction + shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_batch = self.depth_transforms(img).to(device)
            with torch.no_grad():
                prediction = self.depth_model(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()  # h,w

            scale = 0.000305
            shift = 0.1378
            depth = scale * prediction + shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
        return depth

    def points_c2w(self, points, P_c2w):
        """
        transform points in camera coord to world coord
        :param points: N,3
        :param P: 4,4
        :return: N,3
        """
        N = points.shape[0]
        points_homogeneous = np.hstack((points, np.ones((N, 1))))

        points_world_homogeneous = points_homogeneous @ P_c2w.T  # (P·A.T).T = A·P.T

        points_world = points_world_homogeneous[:, :3] / points_world_homogeneous[:, 3][:, np.newaxis]
        return points_world

    def depth_K_to_pcd(self, depth_tensor, K, image_np, idx, P_c2w=None, down_sample=False):
        intr_mat_tensor = torch.from_numpy(K).float().to(depth_tensor.device)
        pts = depth_to_3d(depth_tensor[None, None],
                          intr_mat_tensor[None],
                          normalize_points=False)

        points = pts[0].permute(1, 2, 0).cpu().numpy().reshape(-1, 3)
        # pts = depth_to_3d_v2(depth_tensor[None, None],
        #                   intr_mat_tensor,
        #                   normalize_points=False)
        #
        # points = pts[0].permute(1, 2, 0).cpu().numpy().reshape(-1, 3)

        if P_c2w is not None:
            points = self.points_c2w(points, P_c2w)
            print('transform points from camera to world')

        pcd_data = o3d.geometry.PointCloud()
        pcd_data.points = o3d.utility.Vector3dVector(points)
        pcd_data.colors = o3d.utility.Vector3dVector(image_np.reshape(-1, 3))
        pcd_data.estimate_normals()
        if down_sample:
            voxel_size = 0.01
            while len(pcd_data.points) > 1_000_000:
                pcd_data = pcd_data.voxel_down_sample(voxel_size=voxel_size)
                voxel_size *= 5

        colors = np.asarray(pcd_data.colors, dtype=np.float32)
        points = np.asarray(pcd_data.points, dtype=np.float32)
        normals = np.asarray(pcd_data.normals, dtype=np.float32)
        pcd = BasicPointCloud(points, colors, normals)
        if idx not in self.mono_pcd:
            self.mono_pcd[idx] = pcd

        return pcd
