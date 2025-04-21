from coordEngine.read_colmap import read_colmap
from coordEngine.read_waymo import read_waymo
from coordEngine.read_nerf_syn import read_nerf_syn

def colmap_demo():
    # colmap_path = 'colmap_cam_demo/waymo_10061/exhaustive'
    colmap_path = 'colmap_cam_demo/garden'
    all_cams = read_colmap(colmap_path=colmap_path)
    print(f'read colmap cameras, number : {len(all_cams)}')
    for cam in all_cams:
        print(cam.image_path)
    all_cams.save_to_json(target_path=colmap_path)

def waymo_demo():
    waymo_path = 'waymo_cam_demo/individual_files_training_segment-10061305430875486848_1080_000_1100_000_with_camera_labels'
    all_cams = read_waymo(waymo_path=waymo_path)
    print(f'read colmap cameras, number : {len(all_cams)}')
    all_cams.save_to_json(target_path=waymo_path)

def nerf_syn_demo():
    nerf_syn_path = 'nerf_cam_demo/nerf_synthetic/nerf_synthetic/chair'
    all_cams = read_nerf_syn(nerf_syn_path=nerf_syn_path)
    print(f'read nerf syn cameras, number : {len(all_cams)}')
    all_cams.save_to_json(target_path=nerf_syn_path)


if __name__ == "__main__":
    colmap_demo()
    # waymo_demo()
    # nerf_syn_demo()