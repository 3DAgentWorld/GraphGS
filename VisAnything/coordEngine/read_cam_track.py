from coordEngine.util_camera import visAnyCameraList, visAnyCamera


def read_cam_track(track_dir, track_name='camera_track_opencv.json', rend_type='3dgs'):
    all_cams = visAnyCameraList()
    all_cams.read_from_json(json_path=track_dir, json_name=track_name)
    if rend_type == '3dgs':
        return all_cams, all_cams.to_3dgs_cameralist()
    elif rend_type == '4dgs':
        return all_cams, all_cams.to_4dgs_cameralist()
    else:
        raise Exception('no implementation of rendering camera type')
