import os
import random
import shutil

def random_select_images(input_folder, output_folder, n):
    """
    随机抽取文件夹中的1/n图像并保存到新的文件夹
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param n: 抽取比例，1/n
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # 计算需要抽取的图像数量
    num_images_to_select = len(image_files) // n

    # 随机抽取图像
    selected_images = random.sample(image_files, num_images_to_select)

    # 复制选中的图像到输出文件夹
    for image in selected_images:
        src_path = os.path.join(input_folder, image)
        dst_path = os.path.join(output_folder, image)
        shutil.copy(src_path, dst_path)

    print(f"sample from {input_folder}, num: {num_images_to_select}/{len(image_files)}, saving to {output_folder}")

# 示例使用
if __name__ == "__main__":
    n = 5  # 抽取比例，1/n
    input_folder = 'datasets/cam_dist/bicycle_camdis/images_r4_s1.2'  # 输入文件夹路径
    output_folder = f'{input_folder}_sample_x{n}/input'  # 输出文件夹路径


    # 随机抽取图像并保存
    random_select_images(input_folder, output_folder, n)
