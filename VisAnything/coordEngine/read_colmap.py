import sys
from VisAnything.coordEngine.colmap_utils import *
from VisAnything.coordEngine.util_camera import visAnyCameraList, visAnyCamera
from VisAnything.coordEngine.util_trans import *


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    all_cams = visAnyCameraList()
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        # ipdb.set_trace()
        height = intr.height
        width = intr.width

        uid = intr.id

        R_c2w = np.transpose(qvec2rotmat(extr.qvec))  # actually this is  cam to world matrix !!! w2c-->transpose-->c2w
        T_w2c = np.array(extr.tvec)  # w2c

        P_w2c = get_4x4(np.transpose(R_c2w), T_w2c)
        P_c2w = w2c_to_c2w(P_w2c)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        K = get_K_from_focal(focal_length_x, focal_length_y, width, height)

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]  # xxx.png--> xxx

        if not os.path.exists(image_path) or "sky_mask" in image_path:
            print("skip =====", image_path)
            continue
        new_image_path = image_path.replace(images_folder, "./input")
        # image = copy.deepcopy(Image.open(image_path))
        cam_info = visAnyCamera(P_c2w=P_c2w, K=K, FovY=FovY, FovX=FovX,
                                image_path=new_image_path, image_name=image_name, width=width, height=height)
        all_cams.add_camera(cam_info)
    sys.stdout.write('\n')
    return all_cams


def read_colmap(colmap_path, from_json=False, images='images'):
    if from_json:
        all_cams = visAnyCameraList()
        all_cams.read_from_json(colmap_path)
        return all_cams

    try:
        cameras_extrinsic_file = os.path.join(colmap_path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(colmap_path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(colmap_path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(colmap_path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    all_cams = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                 images_folder=os.path.join(colmap_path, images))
    all_cams.sort_cam()
    # set sparse point cloud
    colmap_bin2ply(colmap_path)
    sparse_pc_path = os.path.join(colmap_path, "sparse/0/points3D.ply")
    all_cams.set_sparse_pc_path(sparse_pc_path)
    return all_cams


