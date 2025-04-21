import os
import numpy as np
import cv2
import random

def crop_and_resize_image(image, s, r, n, distribution_factor=0.5):
    """
    裁剪并调整图像大小
    :param image: 输入图像
    :param s: 裁剪因子，可以是小数
    :param r: 调整分辨率因子
    :param n: 裁剪数量
    :param distribution_factor: 控制采样分布的因子，0.0表示完全随机，1.0表示完全集中在中心
    :return: 裁剪并调整大小后的图像列表
    """
    H, W, _ = image.shape
    crop_h, crop_w = int(H / s), int(W / s)
    resize_h, resize_w = int(H / r), int(W / r)
    crops = []

    for _ in range(n):
        # 计算中心点
        center_x = int(W / 2 + distribution_factor * (random.randint(-W // 2, W // 2)))
        center_y = int(H / 2 + distribution_factor * (random.randint(-H // 2, H // 2)))

        # 确保中心点在图像范围内
        center_x = max(crop_w // 2, min(center_x, W - crop_w // 2))
        center_y = max(crop_h // 2, min(center_y, H - crop_h // 2))

        # 计算裁剪区域
        x1 = center_x - crop_w // 2
        y1 = center_y - crop_h // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # 裁剪图像
        crop = image[y1:y2, x1:x2]

        # 调整图像大小
        resized_crop = cv2.resize(crop, (resize_w, resize_h))
        crops.append(resized_crop)

    return crops

def process_images(input_folder, output_folder, s, r, n, distribution_factor=0.75):
    """
    处理文件夹中的所有图像
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param s: 裁剪因子，可以是小数
    :param r: 调整分辨率因子
    :param n: 裁剪数量
    :param distribution_factor: 控制采样分布的因子
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # 裁剪并调整图像大小
            cropped_resized_images = crop_and_resize_image(image, s, r, n, distribution_factor)

            # 保存裁剪并调整大小后的图像
            base_filename, ext = os.path.splitext(filename)
            for i, cropped_resized_image in enumerate(cropped_resized_images):
                output_path = os.path.join(output_folder, f"{base_filename}_crop_{i}{ext}")
                cv2.imwrite(output_path, cropped_resized_image)

            print(f"processed {filename} and saving to {output_folder}")

# 示例使用
if __name__ == "__main__":

    s = 1.2  # 裁剪因子，可以是小数
    r = 4  # 调整分辨率因子
    n = 5  # 每张图像裁剪数量
    input_folder = 'datasets/cam_dist/bicycle_camdis/images'  # 输入文件夹路径
    output_folder = f'{input_folder}_r{r}_s{s}'  # 输出文件夹路径
    # 处理图像
    process_images(input_folder, output_folder, s, r, n)
