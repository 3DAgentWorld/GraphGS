import numpy as np
import math


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def get_4x4(R, T):
    """
    :param R: 3x3
    :param T: 3,
    :return:  4x4
    """
    full_matrix = np.eye(4)
    full_matrix[:3, :3] = R
    full_matrix[:3, 3] = T
    return full_matrix


def get_K_from_fov(fovx, fovy, width, height):
    """
    :param fovx:
    :param fovy:
    :param width:
    :param height:
    :return: K, 3x3
    """
    full_matrix = np.eye(3)
    focal_x = fov2focal(fovx, width)
    focal_y = fov2focal(fovy, height)
    full_matrix[0, 0] = focal_x
    full_matrix[1, 1] = focal_y
    full_matrix[0, 2] = width/2
    full_matrix[1, 2] = height/2
    return full_matrix

def get_K_from_focal(focal_x, focal_y, width, height):
    """
    :param focal_x:
    :param focal_y:
    :param width:
    :param height:
    :return: K, 3x3
    """
    full_matrix = np.eye(3)
    full_matrix[0, 0] = focal_x
    full_matrix[1, 1] = focal_y
    full_matrix[0, 2] = width / 2
    full_matrix[1, 2] = height / 2
    return full_matrix

def get_RT(P):
    """
    :param P: 4x4
    :return: R: 3x3 T 3,
    """
    R = P[:3, :3]
    T = P[:3, 3]
    return R, T


def c2w_to_w2c(P):
    """
    :param P: 4x4
    :return: 4x4
    """
    return np.linalg.inv(P)


def w2c_to_c2w(P):
    """
    :param P: 4x4
    :return: 4x4
    """
    return np.linalg.inv(P)


def cv_2_gl(P):
    """
    :param P: 4x4
    :return:
    """
    cv_to_gl = np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, -1], [1, 1, 1, 1]])
    return cv_to_gl * P


def gl_2_cv(P):
    """
    :param P: 4x4
    :return:
    """
    gl_to_cv = np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, -1], [1, 1, 1, 1]])
    return gl_to_cv * P
