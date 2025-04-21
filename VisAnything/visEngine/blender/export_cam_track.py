import bpy
import json
import numpy as np

def export_track(output_filename):

    # 获取场景的设置
    scene = bpy.context.scene
    camera = bpy.context.scene.camera

    # 从相机对象中获取相机内参和图像宽高
    image_width = scene.render.resolution_x
    image_height = scene.render.resolution_y
    # 获取视场角（相机角度）
    angle_x = camera.data.angle_x  # 水平视场角（弧度）
    angle_y = camera.data.angle_y  # 垂直视场角（弧度）

    # 计算内参矩阵 K
    K = np.array([
        [image_width / (2 * np.tan(angle_x / 2)), 0, image_width / 2],
        [0, image_height / (2 * np.tan(angle_y / 2)), image_height / 2],
        [0, 0, 1]
    ])

    # 用于保存相机Pose数据的列表
    camera_poses_list = []
    gl_to_cv = np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, -1], [1, 1, 1, 1]])
    # 遍历时间轴上的每一帧
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)

        # 获取相机的世界矩阵
        P_c2w = np.array(camera.matrix_world)

        # blender sys --> opencv sys
        P_c2w = P_c2w*gl_to_cv
        # 构造输出数据
        camera_data = {
            "P_c2w": P_c2w.tolist(),  # 4x4矩阵展平为列表
            "K": K.tolist(),  # 3x3矩阵展平为列表
            "width": image_width,
            "height": image_height,
            "FovX": angle_x,
            "FovY": angle_y,
            "image_path": f"./track_path/track_{frame:03d}.png",
            "image_name": f"track_{frame:03d}",
        }

        # 将当前帧的相机数据添加到列表
        camera_poses_list.append(camera_data)

    with open(output_filename, 'w') as json_file:
        json.dump(camera_poses_list, json_file, indent=4)

    print(f"相机位姿数据已保存到 {output_filename}")


if __name__ == "__main__":
    # you need to activate camera on track before run this script
    output_filename = r"D:\server\home\songgaochao\codes\visAnything_a6k\colmap_cam_demo\garden\camera_track_opencv.json"
    export_track(output_filename)

