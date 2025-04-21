import bpy
import json
import numpy as np
from mathutils import Matrix, Vector
import os


def load_camera_info_from_json(file_path):
    """
    从 JSON 文件中加载相机信息列表

    参数:
    file_path: str - JSON 文件的路径
    """
    with open(file_path, 'r') as f:
        camera_info_list = json.load(f)
    return camera_info_list


def clear_existing_objects():
    """
    清空现有的所有对象
    """
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def clear_existing_cameras():
    """
    清空现有的所有摄像机对象
    """
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'CAMERA':
            obj.select_set(True)
    bpy.ops.object.delete()


def create_camera(camera_info):
    """
    在 Blender 中创建摄像机对象并设置其参数

    参数:
    camera_info: dict - 包含相机信息的字典
    """
    # 创建摄像机对象
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = f"Camera_{camera_info['image_name']}"

    # 设置摄像机的旋转矩阵和平移向量
    P = np.array(camera_info['P_c2w'])

    # 创建一个4x4的变换矩阵, transfrom opencv to opengl
    transform_matrix = Matrix((
        (P[0, 0], -P[0, 1], -P[0, 2], P[0, 3]),
        (-P[1, 0], P[1, 1], P[1, 2], -P[1, 3]),
        (-P[2, 0], P[2, 1], P[2, 2], -P[2, 3]),
        (0, 0, 0, 1)
    ))

    # 应用变换矩阵到相机对象
    camera.matrix_world = transform_matrix

    # 设置摄像机的视场角（直接使用弧度）
    camera.data.angle_y = camera_info['FovY'] # actually this line is useless
    camera.data.angle_x = camera_info['FovX']


    # 设置分辨率
    bpy.context.scene.render.resolution_x = camera_info['width']
    bpy.context.scene.render.resolution_y = camera_info['height']


def import_cameras_from_json(file_path):
    """
    从 JSON 文件中导入相机并创建 Blender 摄像机对象

    参数:
    file_path: str - JSON 文件的路径
    """
    # 清空现有的摄像机对象
    clear_existing_cameras()

    # 加载相机信息并创建相机对象
    camera_info_list = load_camera_info_from_json(file_path)
    for camera_info in camera_info_list:
        create_camera(camera_info)


def render_ply_with_cameras(json_file_path, ply_file_path, output_dir):
    """
    根据相机参数渲染 PLY 文件图像并导出

    参数:
    json_file_path: str - 相机 JSON 文件的路径
    ply_file_path: str - PLY 文件的路径
    output_dir: str - 渲染图像导出路径
    """
    clear_existing_objects()
    # 导入相机
    print('importing cameras ...')
    import_cameras_from_json(json_file_path)


if __name__ == "__main__":
    json_name = "camera_info_opencv.json"
    json_file_dir = r"D:\server\home\songgaochao\codes\gof_a6k\datasets\TNT_GOF\TrainingSet\Barn"
    json_file_path = os.path.join(json_file_dir, json_name)

    render_ply_with_cameras(json_file_path, None, None)
