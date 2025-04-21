from coordEngine import read_colmap
from coordEngine import camera_trainer

if __name__ == "__main__":
    colmap_path = 'colmap_cam_demo/waymo_10061/exhaustive'
    # colmap_path = 'D:\server\home\songgaochao\codes\gof_a6k\datasets\TNT_GOF\TrainingSet\Barn'
    all_cams = read_colmap.read_colmap(colmap_path=colmap_path)
    print(f'read colmap cameras, number : {len(all_cams)}')
    all_cams.save_to_json(target_path=colmap_path)

    depth_pred = camera_trainer.DepthPredictor(all_cams)
    # depth_pred.generate_one_pcd(idx=0, data_dir=colmap_path)
    depth_pred.generate_multi_pcd(frames=[0, 24, 1], data_dir=colmap_path)

