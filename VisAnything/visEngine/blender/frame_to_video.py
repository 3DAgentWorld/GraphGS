import os
import cv2


def create_video_from_images(folder_path):
    # 获取文件夹名
    folder_name = os.path.basename(folder_path)
    video_path = os.path.join(folder_path, f"{folder_name}.mp4")

    # 获取所有图像文件，支持 jpg 和 png 格式
    images = [img for img in os.listdir(folder_path) if img.endswith(('.jpg', '.png'))]

    # 检查是否有图像
    if not images:
        print("该文件夹中没有找到图像文件。")
        return

        # 根据文件名排序
    images.sort()

    # 读取第一张图像，获取宽度和高度
    first_image = cv2.imread(os.path.join(folder_path, images[0]))
    height, width, layers = first_image.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))  # 30帧每秒

    # 循环读取图像并写入视频
    for image in images:
        img_path = os.path.join(folder_path, image)
        print(f'reading: {img_path}')
        img = cv2.imread(img_path)
        video.write(img)

        # 释放视频写入对象
    video.release()
    print(f"视频已保存至: {video_path}")


if __name__ == "__main__":
    folder_path = input("请输入文件夹路径: ")
    create_video_from_images(folder_path)