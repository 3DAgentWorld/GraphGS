import cv2
import os

# 定义视频文件夹和输出帧文件夹
videos_folder = r'D:\server\home\songgaochao\datasets_a6000\flame_steak\flame_steak'
output_folder = r'D:\server\home\songgaochao\datasets_a6000\flame_steak\flame_steak_multi'

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历视频文件夹中的所有文件
for filename in os.listdir(videos_folder):
    if filename.endswith('.mp4'):
        # 构建视频文件的完整路径
        video_path = os.path.join(videos_folder, filename)

        # 创建存储该视频帧的文件夹
        video_frames_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
        print(f'mp4 file folder {video_frames_folder}')
        if not os.path.exists(video_frames_folder):
            os.makedirs(video_frames_folder)

            # 读取视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            continue
        frame_count = 0

        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                break

                # 构建帧文件的完整路径
            # frame_filename = os.path.join(video_frames_folder, f'images')
            # os.makedirs(frame_filename, exist_ok=True)
            frame_filename = video_frames_folder
            frame_filename = os.path.join(frame_filename, f'frame_{frame_count:04d}.jpg')
            print(f'frame: {frame_filename}')
            # 保存帧
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

            # 释放视频捕获对象
        cap.release()

print("所有视频帧已成功导出！")
