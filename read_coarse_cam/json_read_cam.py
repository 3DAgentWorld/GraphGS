import os
from VisAnything.coordEngine.util_camera import visAnyCameraList, visAnyCamera
import numpy as np
import math
from typing import NamedTuple

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
    def todict(self):
        return {
            'uid': self.uid,
            'image_path': self.image_name,
            'image_name': self.image_name,
            'new_name': self.new_name,
            'width': self.width,
            'height': self.height,
            'camera_id': self.camera_id
        }


def readCoarsecamJson(dataset_path):
    all_cams = visAnyCameraList()
    all_cams.read_from_json(json_path=dataset_path, json_name='camera_info_opencv.json')
    cam_infos=[]
    for ind, cam in enumerate(all_cams):
        R = cam.P_c2w[:3, :3] 
        T = cam.P_c2w[:3, 3].reshape(1, 3)
        D = cam.P_c2w[:3, 2].reshape(1, 3)
        # 获取图片路径的格式后缀
        image_format = os.path.splitext(cam.image_path)[1]  # 例如 .jpg
        cam_info=CameraInfo(uid=ind, R=R, T=T, D=D, FovY=cam.FovY, FovX=cam.FovX, image_path=os.path.join(dataset_path,"input",cam.image_name, image_format), image_name=(cam.image_name + image_format), width=cam.width, height=cam.height, camera_id=None, new_name=(cam.image_name + image_format))
        cam_infos.append(cam_info)
    return cam_infos
