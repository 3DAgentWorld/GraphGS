# VisAnything
![main architecture](./pageImages/camMatrix3.drawio_00.png)

# 1. Support free camera movement !!!

## Camera move demos

<div style="display: flex; justify-content: space-around;">  

  <div style="text-align: center;">  
    <img src="pageImages/cam_track_demo/garden_rend_blender.gif" alt="First GIF" style="width: 400px;"/>  
    <p>Blender Mesh</p>  
  </div>  

  <div style="text-align: center;">  
    <img src="pageImages/cam_track_demo/garden_rend_n_blender.gif" alt="Second GIF" style="width: 400px;"/>  
    <p>Blender Normal</p>  
  </div>  

  <div style="text-align: center;">  
    <img src="pageImages/cam_track_demo/garden_rend_gs.gif" alt="Third GIF" style="width: 400px;"/>  
    <p>3d Gaussian NVS</p>  
  </div>  

</div>

<div style="display: flex; justify-content: space-around;">  

  <div style="text-align: center;">  
    <img src="pageImages/cam_track_demo/flame_steak_points_blender.gif" alt="xx GIF" style="width: 400px;"/>  
    <p>Blender Points</p>  
  </div>  

  <div style="text-align: center;">  
    <img src="pageImages/cam_track_demo/flame_steak_rend_gs.gif" alt="xxx GIF" style="width: 400px;"/>  
    <p>4d Gaussian NVS</p>  
  </div>

</div>

## Pipeline
1. Import mesh or GS points in blender, and set camera track (see `setCamTrackInBlender.md`)
2. Run `visEngine/blender/export_cam_track.py`, to export cameras for all frames. These are
stored in `camera_track_opencv.json`
3. Read `camera_track_opencv.json` in any project, please refer to `demo_track_read.py`


# 2. Environments
install the following packages

1. basic
- plyfile: for managing point cloud
- numpy
- skimage

2. ace visualization
- torch
- trimesh: for meshing camera traj
- pyrender: for rendering
- Pillow: for PIL Image
- scipy
- matplotlib
- skimage
- bisect

3. mono depth
- torch
- kornia: for mono depth to pcd
- Pillow: for PIL Image
- opencv-python: for cv2
- open3d: for point clouds

# 3. Camera file config
cameras are stored in `camera_info_opencv.json` with following formats:
- `P_c2w`: camera to world matrix in opencv/colmap system, 4x4
- `K`: camera intrinsic parameter, 3x3
- `width`: image width
- `height`: image height
- `FovX`: fov x, calculated from K
- `FovY`: fov y, calculated from K
- `image_path`: relative path of image
- `image_name`: image name without .png/.jpg

