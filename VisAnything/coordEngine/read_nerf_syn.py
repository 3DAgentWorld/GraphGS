from VisAnything.coordEngine.util_trans import *
import os
import json
from pathlib import Path
from PIL import Image
from VisAnything.coordEngine.util_camera import visAnyCameraList, visAnyCamera
from VisAnything.coordEngine import colmap_utils


def readNerfSynCameras(path, transformsfile, white_background, extension=".png"):
    all_cams = visAnyCameraList()
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            image_path = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # correct transform ?
            P_c2w = c2w
            P_c2w = gl_2_cv(P_c2w)

            # # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1
            #
            # # get the world-to-camera transform and set R, T
            # w2c = np.linalg.inv(c2w)
            # R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            # T = w2c[:3, 3]
            #
            # # suppose opencv origin
            # R_c2w = R
            # T_w2c = T
            # P_w2c = get_4x4(np.transpose(R_c2w), T_w2c)
            # P_c2w = w2c_to_c2w(P_w2c)
            ###

            image_name = Path(image_path).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            width, height = image.size
            focal_length_x = fov2focal(fovx, width)
            fovy = focal2fov(focal_length_x, height)
            FovY = fovy
            FovX = fovx

            focal_length_y = focal_length_x

            K = get_K_from_focal(focal_length_x, focal_length_y, width, height)
            new_image_path = image_path.replace(path, "./input")
            cam_info = visAnyCamera(P_c2w=P_c2w, K=K, FovY=FovY, FovX=FovX,
                                    image_path=new_image_path, image_name=image_name, width=width, height=height)

            all_cams.add_camera(cam_info)

    return all_cams


def read_nerf_syn(nerf_syn_path, white_background=False, add_test=False, extension=".png"):
    print("Reading transforms_train.json")
    train_cams = readNerfSynCameras(nerf_syn_path, "transforms_train.json", white_background, extension)
    print("Reading transforms_test.json")
    test_cams = readNerfSynCameras(nerf_syn_path, "transforms_test.json", white_background, extension)

    if add_test:
        train_cams.all_camera_list.extend(test_cams.all_camera_list)

    ply_path = os.path.join(nerf_syn_path, "points3d.ply")
    pcd = colmap_utils.generate_random_pc(ply_path)
    if os.path.exists(ply_path):
        train_cams.sparse_pc_path = ply_path

    return train_cams
