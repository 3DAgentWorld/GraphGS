from coordEngine.read_cam_track import read_cam_track


def read_track_demo():
    track_dir = 'demos_dataset/colmap_cam_demo/garden'
    track_name = 'camera_track_opencv.json'
    all_cams, all_cams_rend = read_cam_track(track_dir=track_dir, track_name=track_name, rend_type='4dgs')
    for cam, cam_r in zip(all_cams, all_cams_rend):
        print(cam.image_path, cam_r.full_proj_transform.shape)


if __name__ == "__main__":
    read_track_demo()
