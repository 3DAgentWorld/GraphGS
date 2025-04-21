import os
import pickle
import sys
import numpy as np
from typing import NamedTuple
import math
import shutil
# from noise_camera import noise_cam


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    D: np.array
    FovY: np.array
    FovX: np.array
    # image: np.array
    new_name: str
    image_path: str
    image_name: str
    width: int
    height: int
    camera_id: str


def get_prefix(camera_id):
    if camera_id == "camera_FRONT":
        prefix = 0
    elif camera_id == "camera_FRONT_LEFT":
        prefix = 1
    elif camera_id == "camera_FRONT_RIGHT":
        prefix = 2
    else:
        raise Exception()
    return prefix


def readWaymoCameras(observers_data, images_folder, start_end=None):
    cam_infos = []
    counter = 0
    for camera_id, camera_data in observers_data.items():
        if 'camera' in camera_id and 'SIDE' not in camera_id:
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write("Reading camera {}/{}, id={}\n".format(1, len(observers_data), camera_id))
            sys.stdout.flush()
            prefix = get_prefix(camera_id)
            it_range = range(camera_data['n_frames'])
            if start_end is not None:
                it_range = range(start_end[0], start_end[1])
            for idx in it_range:
                transform_matrix = np.array(camera_data['data']['c2w'][idx])
                intrinsic_matrix = np.array(camera_data['data']['intr'][idx])
                hw = np.array(camera_data['data']['hw'][idx])

                extr = transform_matrix
                intr = intrinsic_matrix
                height = hw[0]
                width = hw[1]

                waymo_to_colmap = np.array([
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]
                ])

                transform_matrix = waymo_to_colmap @ transform_matrix
                uid = counter
                counter += 1
                # transform_matrix = noise_cam(transform_matrix)
                R = transform_matrix[:3, :3] # actually is c2w
                T = transform_matrix[:3, 3].reshape(1, 3)
                D = transform_matrix[:3, 2].reshape(1, 3)

                fx = intr[0, 0]
                fy = intr[1, 1]
                cx = intr[0, 2]
                cy = intr[1, 2]

                FovY = focal2fov(fy, height)
                FovX = focal2fov(fx, width)

                image_path = os.path.join(images_folder, camera_id, f"{idx:08d}.jpg")
                image_name = os.path.basename(image_path)
                # image = Image.open(image_path)
                new_name = f'{prefix}_{image_name}'

                cam_info = CameraInfo(uid=uid, R=R, T=T, D=D, FovY=FovY, FovX=FovX,
                                      image_path=image_path, image_name=image_name, width=width,
                                      height=height, camera_id=camera_id, new_name=new_name)
                cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readWaymoSceneInfo(path, images=None, start_end=None):
    file_path = os.path.join(path, 'scenario.pt')

    # 以二进制读取模式打开文件
    with open(file_path, 'rb') as file:
        # 使用pickle的load函数载入数据
        data = pickle.load(file)

    observers_data = data["observers"]
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readWaymoCameras(observers_data=observers_data, images_folder=os.path.join(path, reading_dir),
                                          start_end=start_end)
    # cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    return cam_infos_unsorted


def merge_save_waymo(dataset_folder, start_end=None):
    source_folders_i = ['camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT']
    source_folders = [os.path.join(dataset_folder, 'images', x) for x in source_folders_i]
    # 目标文件夹
    if start_end is not None:
        destination_folder = os.path.join(dataset_folder, f'frame_{start_end[0]}-{start_end[1]}', 'input')
    else:
        destination_folder = os.path.join(dataset_folder, 'input')
    if os.path.exists(destination_folder):
        print(f'{destination_folder} exists, jump')
        return os.path.dirname(destination_folder)

    print(f'not find input folder, merge saving to {destination_folder}')
    os.makedirs(destination_folder)
    # 遍历源文件夹
    if start_end is not None:
        for ind, folder in enumerate(source_folders):
            for idx in range(start_end[0], start_end[1]):
                filename = f"{idx:08d}.jpg"
                source_path = os.path.join(folder, filename)
                new_filename = f'{get_prefix(source_folders_i[ind])}_{filename}'
                destination_path = os.path.join(destination_folder, new_filename)  # 保留原始文件名
                shutil.copy(source_path, destination_path)
    else:
        for ind, folder in enumerate(source_folders):
            for filename in os.listdir(folder):
                if filename.upper().endswith('.JPG') or filename.upper().endswith('.PNG'):  # 假设你只处理 jpg 和 png 文件
                    source_path = os.path.join(folder, filename)
                    new_filename = f'{get_prefix(source_folders_i[ind])}_{filename}'
                    destination_path = os.path.join(destination_folder, new_filename)  # 保留原始文件名
                    shutil.copy(source_path, destination_path)
    return os.path.dirname(destination_folder)


def get_prefix_2(camera_id):
    if camera_id == "image_0":
        prefix = 0
    elif camera_id == "image_1":
        prefix = 1
    elif camera_id == "image_2":
        prefix = 2
    else:
        raise Exception()
    return prefix


def merge_save_partWaymo(dataset_folder, dataset_id, target_folder):
    source_folders_i = ['image_0', 'image_1', 'image_2']
    source_folders = [os.path.join(dataset_folder, dataset_id, x) for x in source_folders_i]
    # 目标文件夹
    destination_folder = os.path.join(target_folder, dataset_id, 'input')

    os.makedirs(destination_folder, exist_ok=True)
    # 遍历源文件夹
    for ind, folder in enumerate(source_folders):
        for filename in os.listdir(folder):
            if filename.upper().endswith('.JPG') or filename.upper().endswith('.PNG'):  # 假设你只处理 jpg 和 png 文件
                source_path = os.path.join(folder, filename)
                new_filename = f'{get_prefix_2(source_folders_i[ind])}_{filename}'
                destination_path = os.path.join(destination_folder, new_filename)  # 保留原始文件名
                shutil.copy(source_path, destination_path)


if __name__ == "__main__":
    dataset_folder = "waymo_new/waymo_scenes"
    # merge_save_waymo(pt_folder)
    merge_save_partWaymo(dataset_folder, '0158150', './own_data/waymo_new')
