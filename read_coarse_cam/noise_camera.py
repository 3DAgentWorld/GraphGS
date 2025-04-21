import numpy as np


def noise_cam(R, theta_x=0.6, theta_y=0.6, delta_pos=0.1):
    # 原始的相机到世界矩阵 R (4x4)

    # 生成随机的微小旋转角度
    theta_x = np.random.uniform(-theta_x, theta_x)  # 在[-0.1, 0.1]范围内生成随机角度
    theta_y = np.random.uniform(-theta_y, theta_y)  # 在[-0.1, 0.1]范围内生成随机角度

    # 生成随机的微小位移
    dx, dy, dz = np.random.uniform(-delta_pos, delta_pos, 3)  # 在[-0.1, 0.1]范围内生成随机位移

    # 生成随机的旋转矩阵
    delta_R_x = np.array([[1, 0, 0, 0],
                          [0, np.cos(theta_x), -np.sin(theta_x), 0],
                          [0, np.sin(theta_x), np.cos(theta_x), 0],
                          [0, 0, 0, 1]])

    delta_R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y), 0],
                          [0, 1, 0, 0],
                          [-np.sin(theta_y), 0, np.cos(theta_y), 0],
                          [0, 0, 0, 1]])

    # 计算分别绕x轴和y轴旋转后的相机到世界矩阵
    R_x = np.dot(R, delta_R_x)
    R_xy = np.dot(R_x, delta_R_y)

    # 对相机的世界坐标进行随机微小扰动
    T = np.array([[1, 0, 0, dx],
                  [0, 1, 0, dy],
                  [0, 0, 1, dz],
                  [0, 0, 0, 1]])

    # 应用随机微小位移得到最终的相机到世界矩阵
    R_final = np.dot(R_xy, T)
    return R_final
