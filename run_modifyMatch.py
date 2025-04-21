import os
from argparse import ArgumentParser
import shutil
import subprocess
from match_script import generate_match_pairs
from read_coarse_cam.waymo_read_cam import merge_save_waymo
import datetime


def check_ba(dataset_dir, exp_name):
    exp_dir = os.path.join(dataset_dir, exp_name)
    input_dir = os.path.join(dataset_dir, 'input')
    ba_dir = os.path.join(exp_dir, 'images')
    generated_num = len(os.listdir(ba_dir))
    input_num = len(os.listdir(input_dir))
    if_pass = True
    if generated_num < input_num:
        if_pass = False
    with open(os.path.join(dataset_dir, 'check_log.txt'), 'a') as f:
        f.write(f'{datetime.datetime.now()}\n')
        f.write(f'check {exp_name}: input/images={input_num}/{generated_num}, pass={if_pass}\n\n')
    return if_pass


parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--source_path", "-s",  
                    type=str)
parser.add_argument("--camera", default="SIMPLE_PINHOLE", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--dataset_type", default="ready", type=str)
parser.add_argument("--match_mode", default='E', type=str)
# O_neighbor, O_neighborJump, O_jump (these 3 modes have no cam) | C (just concentric match)
#      | Q (quadrant filter + concentric match) | E (exhaustive) | T (vocab tree)
parser.add_argument("--arg_r", default=5, type=int)
parser.add_argument("--arg_k", default=10, type=int)
parser.add_argument("--arg_w", default=1, type=int)
parser.add_argument("--arg_s", default=False, type=bool)  # strict mode of Q
parser.add_argument("--threads", default=32, type=int)
parser.add_argument("--gpu_id", default=0, type=int)

waymo_frame = [0, 50]
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
use_gpu = 1 if not args.no_gpu else 0
vocab_tree_path = r"vocab_tree/vocab_tree_flickr100K_words32K.bin"

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    match_mode = args.match_mode
    threads = str(args.threads)
    strict_mode = 'loose'
    if args.arg_s is True:
        strict_mode = 'strict'
    exp_name_dir = {
        'E': 'exhaustive',
        'T': 'vocab_tree',
        'O_neighbor': f'O_neighbor_{args.arg_r}',
        'O_neighborJump': f'O_neighborJump_{args.arg_r}_{args.arg_k}',
        'O_jump': f'O_jump_{args.arg_k}',
        'C': f'concentric_{args.arg_r}_{args.arg_k}_{args.arg_w}',
        'Q': f'Q_concentric_{args.arg_r}_{args.arg_k}_{args.arg_w}_{strict_mode}'
    }
    before_source_path = args.source_path
    if args.dataset_type == "waymo":  # save ['camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT'] to $s$/input
        before_source_path = merge_save_waymo(before_source_path)
    elif args.dataset_type == "waymo-f":  # save ['camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT'] to $s$/input, some frame
        before_source_path = merge_save_waymo(before_source_path, waymo_frame)
    # elif args.dataset_type == 'colmap':  # save $s$/images to $s$/input
    #     merge_save_colamp(before_source_path)
    # elif args.dataset_type == 'mill':
    #     merge_save_mill(before_source_path)
    # elif args.dataset_type.startswith('kitti'):
    #     scene_id = args.dataset_type.split('-')[-3]
    #     start_frame = int(args.dataset_type.split('-')[-2])
    #     end_frame = int(args.dataset_type.split('-')[-1])
    #     before_source_path = merge_save_kitti(before_source_path, scene_id, start_frame, end_frame)
    elif args.dataset_type == 'ready':  # you already prepared $s$/input
        pass
    else:
        raise Exception("dataset type wrong")

    exp_name = exp_name_dir[match_mode]
    exp_path = os.path.join(before_source_path, exp_name)
    if os.path.exists(exp_path):
        user_input = input(f"Find folder already exists: {exp_name}, delete ? 'yes' or 'no': ").strip().lower()
        if user_input == 'yes':
            for filename in os.listdir(exp_path):
                file_path = os.path.join(exp_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # 删除文件或符号链接
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # 删除子文件夹
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            raise Exception(f'{exp_path} exist, please check')

    print(f'creating folder: {exp_path}')
    os.makedirs(os.path.join(exp_path, 'distorted/sparse'), exist_ok=True)
    image_path = os.path.join(before_source_path, 'input')

    ## Feature extraction
    print('feature extracting')
    feat_extracton_cmd = colmap_command + " feature_extractor " \
                                          "--database_path " + os.path.join(exp_path, "distorted/database.db") + " \
        --image_path " + image_path + " \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu) + f" \
        --SiftExtraction.num_threads {threads}"
    with open(os.path.join(exp_path, 'feature_log.txt'), 'w') as f:
        subprocess.run(feat_extracton_cmd, stdout=f, stderr=subprocess.STDOUT, shell=True)

    ## Feature matching
    print('feature matching')
    if exp_name == 'exhaustive':
        print('use exhaustive matching')
        feat_matching_cmd = colmap_command + " exhaustive_matcher \
                --database_path " + os.path.join(exp_path, "distorted/database.db") + " \
                --SiftMatching.use_gpu " + str(use_gpu)
        with open(os.path.join(exp_path, 'match_log.txt'), 'w') as f:
            subprocess.run(feat_matching_cmd, stdout=f, stderr=subprocess.STDOUT, shell=True)
    elif exp_name == 'vocab_tree':
        print('use vocab_tree matching')
        # VOCABTREE_PATH=/usr/local/google/home/bmild/vocab_tree_flickr100K_words32K.bin
        # colmap vocab_tree_matcher \
        #     --database_path "$DATASET_PATH"/database.db \
        #     --VocabTreeMatching.vocab_tree_path $VOCABTREE_PATH \
        #     --SiftMatching.use_gpu "$USE_GPU"

        feat_matching_cmd = colmap_command + " vocab_tree_matcher \
                --database_path " + os.path.join(exp_path, "distorted/database.db") + " \
                --VocabTreeMatching.vocab_tree_path " + vocab_tree_path + " \
                --SiftMatching.use_gpu " + str(use_gpu)
        with open(os.path.join(exp_path, 'match_log.txt'), 'w') as f:
            subprocess.run(feat_matching_cmd, stdout=f, stderr=subprocess.STDOUT, shell=True)
    else:
        print(f'use modified matching, match_mode {match_mode}')
        match_file_path = generate_match_pairs(image_path, match_mode,
                                               arg_list=[args.arg_r, args.arg_k, args.arg_w, args.arg_s],
                                               dataset_path=before_source_path,
                                               exp_name=exp_name, dataset_type=args.dataset_type,
                                               waymo_frame=waymo_frame)
        print(f'match pairs file done, start matching')
        feat_matching_cmd = colmap_command + " matches_importer \
                    --database_path " + os.path.join(exp_path, "distorted/database.db") + " \
                    --match_list_path " + match_file_path + " --match_type pairs" + " \
                    --SiftMatching.use_gpu " + str(use_gpu) + f" \
                    --SiftMatching.num_threads {threads} "
        with open(os.path.join(exp_path, 'match_log.txt'), 'w') as f:
            subprocess.run(feat_matching_cmd, stdout=f, stderr=subprocess.STDOUT, shell=True)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    print('BA ing')
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + os.path.join(exp_path, "distorted/database.db") + " \
        --image_path " + image_path + " \
        --output_path " + os.path.join(exp_path, "distorted/sparse") +
                  " --Mapper.ba_global_function_tolerance=0.000001" +
                  " --Mapper.init_min_tri_angle=4" +
                  " --Mapper.ba_global_max_refinements=1" +
                  " --Mapper.ba_global_images_ratio=2.1" +
                  " --Mapper.ba_global_points_ratio=2.1" +
                  " --Mapper.ba_global_images_freq=1000" +
                  " --Mapper.abs_pose_max_error=12" +  # 12
                  " --Mapper.abs_pose_min_num_inliers=6" +  # 30
                  " --Mapper.abs_pose_min_inlier_ratio=0.05" +  # 0.25
                  " --Mapper.max_reg_trials=3" +  # 3
                  f" --Mapper.num_threads={threads}" +
                  " --Mapper.multiple_models=0")
    with open(os.path.join(exp_path, 'BA_log.txt'), 'w') as f:
        subprocess.run(mapper_cmd, stdout=f, stderr=subprocess.STDOUT, shell=True)

    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    print('undistore ing')
    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + image_path + " \
        --input_path " + os.path.join(exp_path, "distorted/sparse/0") + " \
        --output_path " + exp_path + " \
        --output_type COLMAP")
    with open(os.path.join(exp_path, 'undist_log.txt'), 'w') as f:
        subprocess.run(img_undist_cmd, stdout=f, stderr=subprocess.STDOUT, shell=True)

    os.makedirs(os.path.join(exp_path, 'sparse/0'), exist_ok=True)
    files = os.listdir(os.path.join(exp_path, 'sparse'))

    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(exp_path, "sparse", file)
        destination_file = os.path.join(exp_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    print('check ba ing')
    if_pass = True
    try:
        if_pass=check_ba(dataset_dir=before_source_path, exp_name=exp_name)
    except Exception as e:
        print(f'[WARNING] BA failed!, you may change parameters and rematch!')
        print(f'Error: {e}')
    if not if_pass:
        print('[WARNING] BA failed, you may change parameters and rematch!')
    
    print("Done.")


