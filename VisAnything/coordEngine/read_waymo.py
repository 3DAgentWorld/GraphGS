import os
import pickle
import sys
import shutil
import numpy as np
from VisAnything.coordEngine.util_camera import visAnyCameraList, visAnyCamera
from VisAnything.coordEngine.util_trans import *
from pathlib import Path


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
    all_cams = visAnyCameraList()
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

                extr = transform_matrix # colmap c2w
                intr = intrinsic_matrix
                height = hw[0]
                width = hw[1]

                P_c2w = transform_matrix
                K = intr


                fx = intr[0, 0]
                fy = intr[1, 1]
                # cx = intr[0, 2]
                # cy = intr[1, 2]

                FovY = focal2fov(fy, height)
                FovX = focal2fov(fx, width)

                image_path = os.path.join("./input", camera_id, f"{idx:08d}.jpg")
                image_name = Path(image_path).stem
                # image = Image.open(image_path)
                new_name = f'{prefix}_{image_name}'

                cam_info = visAnyCamera(P_c2w=P_c2w, K=K, FovY=FovY, FovX=FovX,
                                        image_path=image_path, image_name=new_name, width=int(width),
                                        height=int(height))
                # import ipdb;ipdb.set_trace()
                all_cams.add_camera(cam_info)
    sys.stdout.write('\n')
    return all_cams


def read_waymo(waymo_path, images=None, start_end=None):
    file_path = os.path.join(waymo_path, 'scenario.pt')

    # 以二进制读取模式打开文件
    with open(file_path, 'rb') as file:
        # 使用pickle的load函数载入数据
        data = pickle.load(file)

    observers_data = data["observers"]
    reading_dir = "images" if images == None else images
    all_cams = readWaymoCameras(observers_data=observers_data, images_folder=os.path.join(waymo_path, reading_dir),
                                start_end=start_end)
    all_cams.sort_cam()
    return all_cams


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
                if filename.endswith('.jpg') or filename.endswith('.png'):  # 假设你只处理 jpg 和 png 文件
                    source_path = os.path.join(folder, filename)
                    new_filename = f'{get_prefix(source_folders_i[ind])}_{filename}'
                    destination_path = os.path.join(destination_folder, new_filename)  # 保留原始文件名
                    shutil.copy(source_path, destination_path)
    return os.path.dirname(destination_folder)

