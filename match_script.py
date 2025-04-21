import os
import numpy as np
from scipy.spatial import cKDTree
from read_coarse_cam.waymo_read_cam import readWaymoSceneInfo
from read_coarse_cam.json_read_cam import readCoarsecamJson
from VisAnything.coordEngine.util_camera import visAnyCameraList, visAnyCamera
import networkx as nx
import json
def generate_match_pairs(image_path, match_mode, arg_list, dataset_path, exp_name, dataset_type, waymo_frame=None):
    """
    :param dataset_type:
    :param dataset_path:
    :param exp_name:
    :param image_path:  xxxxx/input
    :param match_mode:  O_neighbor, O_neighborJump, O_jump (these 3 modes have no cam) | C (just concentric match)
     | Q (quadrant filter + concentric match) | E (exhaustive)
    :param arg_list: [r=neighbor radius, k=jump num, w=concentric width, s=strict for quadrant filter]
    :param pair_path:  xxxxx/exp_name/match_mode_pairs.txt
    :return:
    """
    match_file_path = os.path.join(dataset_path, exp_name, f'pairs_{exp_name}.txt')
    if dataset_type == 'waymo':
        if match_mode in ['O_neighbor', 'O_neighborJump', 'O_jump']:
            print(f'Original Match: {match_file_path}')
            original_match(image_path, match_mode, arg_list, match_file_path)
        else:
            waymo_match(match_mode, arg_list, dataset_path, match_file_path)
    elif dataset_type == 'waymo-f':
        if match_mode in ['O_neighbor', 'O_neighborJump', 'O_jump']:
            print(f'Original Match: {match_file_path}')
            original_match(image_path, match_mode, arg_list, match_file_path)
        else:
            waymo_match(match_mode, arg_list, dataset_path, match_file_path, waymo_frame)
    elif dataset_type == 'ready':  # no coarse camera can be used
        if match_mode in ['C','Q']:
            print(f'Match mode: {match_mode}, saving results to {match_file_path}')
            CQ_match(match_mode, arg_list, dataset_path, match_file_path)
        else:
            print(f'Original Match: {match_file_path}')
            original_match(image_path, match_mode, arg_list, match_file_path)
    else:
        raise Exception('not implement dataset_type')

    return match_file_path


def waymo_match(match_mode, arg_list, dataset_path, output_folder, waymo_frame=None):
    log_path = output_folder[:-4] + '_log.txt'
    if waymo_frame is not None:
        cam_infos = readWaymoSceneInfo(dataset_path, waymo_frame)
    else:
        cam_infos = readWaymoSceneInfo(dataset_path)
    print("Concentric_matching...")
    match_pairs = concentric_match(cam_infos, r=arg_list[0], k=arg_list[1], w=arg_list[2])
    print(f'concentric pairs num: {len(match_pairs)}')
    intersection = []
    if match_mode == 'Q':
        print("quadrant_matching...")
        match_paris_filter = quadrant_filter_match(cam_infos, s=arg_list[3])
        print(f'quadrant pairs num: {len(match_paris_filter)}')
        print("quadrant_filtering")
        intersection = [x for x in match_pairs if x in match_paris_filter]
    elif match_mode == 'C':
        intersection = match_pairs
    print(f'intersection nums: {len(intersection)}')
    img_nums = len(cam_infos)
    dense_pairs = img_nums * (img_nums - 1) / 2
    print(f'img nums/dense pairs: {img_nums}/{dense_pairs}')
    with open(output_folder, 'w') as f:
        for item in intersection:
            f.write(item[0] + ' ' + item[1] + '\n')
    all_m, unique_m = remove_duplication(input_file=output_folder, output_file=output_folder)
    with open(log_path, 'w') as f:
        f.write(f'concentric pairs num: {len(match_pairs)}\n')
        if match_mode == "Q":
            f.write(f'quadrant pairs num: {len(match_paris_filter)}\n')
        f.write(f'intersection nums: {len(intersection)}\n')
        f.write(f'img nums/dense pairs: {img_nums}/{dense_pairs}\n')
        f.write(f'all pairs/unique pairs: {all_m}/{unique_m}\n')
    return output_folder

def CQ_match(match_mode, arg_list, dataset_path, output_folder, waymo_frame=None):
    log_path = output_folder[:-4] + '_log.txt'
    graph_path = output_folder[:-4] + '_graphinfo.json'
    cam_infos = readCoarsecamJson(dataset_path)
    
    print("Concentric_matching...")
    match_pairs, match_pairs_obj, match_pairs_C_obj = concentric_match(cam_infos, r=arg_list[0], k=arg_list[1], w=arg_list[2])
    print(f'concentric pairs num: {len(match_pairs)}')
    intersection = []
    intersection_obj = []
    if match_mode == 'Q':
        print("quadrant_matching...")
        match_pairs_filter, match_pairs_filter_obj  = quadrant_filter_match(cam_infos, s=arg_list[3])
        print(f'quadrant pairs num: {len(match_pairs_filter)}')
        print("quadrant_filtering")
        # match_pairs_filter=match_pairs_filter[:1000]
        # match_pairs_filter_obj=match_pairs_filter_obj[:1000]
        intersection = [x for x in match_pairs if x in match_pairs_filter]
        intersection_obj = [x_obj for x, x_obj in zip(match_pairs, match_pairs_obj) if x in intersection]
        intersection+=[[pair[0].new_name, pair[1].new_name] for pair in match_pairs_C_obj]
        intersection_obj+=match_pairs_C_obj
    elif match_mode == 'C':
        intersection = match_pairs
    print(f'intersection nums: {len(intersection)}')
    img_nums = len(cam_infos)
    dense_pairs = img_nums * (img_nums - 1) / 2
    print(f'img nums/dense pairs: {img_nums}/{dense_pairs}')
    with open(output_folder, 'w') as f:
        for item in intersection:
            f.write(item[0] + ' ' + item[1] + '\n')
    all_m, unique_m = remove_duplication(input_file=output_folder, output_file=output_folder)
    with open(log_path, 'w') as f:
        f.write(f'concentric pairs num: {len(match_pairs)}\n')
        if match_mode == "Q":
            f.write(f'quadrant pairs num: {len(match_pairs_filter)}\n')
        f.write(f'intersection nums: {len(intersection)}\n')
        f.write(f'img nums/dense pairs: {img_nums}/{dense_pairs}\n')
        f.write(f'all pairs/unique pairs: {all_m}/{unique_m}\n')

    to_graph(cam_infos, intersection_obj, graph_path)
    return output_folder

def to_graph(cam_info_list, cam_pairs_obj, graph_path, k=0.5):
    edges=[]
    for cam_pair in cam_pairs_obj:
        node1=cam_pair[0].uid
        node2=cam_pair[1].uid
        p_1=cam_pair[0].T[0]
        p_2=cam_pair[1].T[0]
        d_1=cam_pair[0].D[0]
        d_2=cam_pair[1].D[0]
        edge_weight=np.exp(-np.linalg.norm(p_1-p_2))/(1-k*np.exp(-np.dot(d_1,d_2)))
        edges.append((node1, node2, edge_weight))
    unique_edges=set(edges)
    G=nx.Graph()
    G.add_weighted_edges_from(unique_edges)
    betweenness=nx.betweenness_centrality(G, weight='weight')
    node_count = G.number_of_nodes()
    if node_count != len(cam_info_list):
        print(f'G node : {node_count}/{len(cam_info_list)}')
        raise Exception('graph failed since not 1 connect component')
    
    # for node, centrality in betweenness.items():
    #     print(f'node {node}, bet {centrality}')
    data = nx.readwrite.json_graph.node_link_data(G)
    json_data = json.dumps(data)
    with open(graph_path, 'w') as json_file:
        json_file.write(json_data)
        print(f'save graph to {graph_path}')

    dict_list = [obj.todict() for obj in cam_info_list]

    # 序列化为 JSON 字符串
    cam_json_data = json.dumps(dict_list)
    cam_path=graph_path.replace('graph','cam')
    with open(cam_path, 'w') as json_file:
        json_file.write(cam_json_data)
        print(f'save coarse caminfo to {cam_path}')



def original_match(image_path, match_mode, arg_list, output_file):
    match_fun_dir = {
        'O_neighbor': neighbor_n_pairs,
        'O_neighborJump': neighbor_jump_nk_pairs,
        'O_jump': jump_k_pairs
    }
    match_function = match_fun_dir[match_mode]
    match_function(image_path, n=arg_list[0], k=arg_list[1], output_file=output_file)
    return output_file


def remove_duplication(input_file, output_file):
    # 用于存储图像匹配
    image_matches = set()
    input_lines = 0
    # 读取输入文件，并处理图像匹配
    with open(input_file, 'r') as file:
        for line in file:
            # 提取每行中的图像名称
            input_lines += 1
            image1, image2 = line.strip().split()
            # 对图像名称进行排序
            sorted_images = tuple(sorted([image1, image2]))
            # 将排序后的图像名称添加到集合中
            image_matches.add(sorted_images)
    print(f'all matches: {input_lines}\n')
    print(f'unique matches: {len(image_matches)}')
    image_matches_list = list(image_matches)
    image_matches_list.sort()
    # 将结果写入新的文件
    with open(output_file, 'w') as file:
        for pair in image_matches_list:
            file.write(' '.join(pair) + '\n')

    return input_lines, len(image_matches)


### original match
def neighbor_n_pairs(folder_path, n=1, k=-1, output_file='default.txt'):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(folder_path):
            files.sort()
            for i in range(len(files) - n):
                for j in range(1, n + 1):
                    f.write(files[i] + ' ' + files[i + j] + '\n')
    return output_file


def jump_k_pairs(folder_path, n=-1, k=2, output_file='default.txt'):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(folder_path):
            files.sort()
            for i in range(len(files)):
                for j in range(i, len(files)):
                    if (j - i) % k == 0 and j != i:
                        f.write(files[i] + ' ' + files[j] + '\n')
    return output_file


def neighbor_jump_nk_pairs(folder_path, n=1, k=2, output_file='default.txt'):
    matching_pairs = []
    matching_pairs_set = set()
    for root, dirs, files in os.walk(folder_path):
        files.sort()
        for i in range(len(files)):
            for j in range(1, n + 1):
                if i + j < len(files):
                    pair = (files[i], files[i + j])
                    if pair not in matching_pairs_set:
                        matching_pairs.append(pair)
                        matching_pairs_set.add(pair)
            for j in range(i, len(files)):
                if (j - i) % k == 0 and j != i:
                    pair = (files[i], files[j])
                    if pair not in matching_pairs_set:
                        matching_pairs.append(pair)
                        matching_pairs_set.add(pair)

    with open(output_file, 'w') as f:
        for pair in matching_pairs:
            f.write(pair[0] + ' ' + pair[1] + '\n')

    return output_file

    # return [cam_1.uid, cam_2.uid, cam_1.camera_id, cam_2.camera_id, cam_1.image_name, cam_2.image_name]
    # return [cam_1.uid, cam_2.uid]


### first step: filter based on quadrant
def quadrant_filter_match(cam_info_list, s=False):
    match_pairs = []
    for i in range(len(cam_info_list)):
        for j in range(len(cam_info_list)):
            cam_i = cam_info_list[i]
            cam_j = cam_info_list[j]
            pos_i = cam_i.T[0]
            pos_j = cam_j.T[0]
            dir_i = cam_i.D[0]
            dir_j = cam_j.D[0]
            pos_quadrant, dir_quadrant = cal_embedding(pos_i, pos_j, dir_i, dir_j)
            if pd_match(pos_quadrant, dir_quadrant, s):
                match_pairs.append(get_match_info(cam_i, cam_j))
                # print([cam_i.uid, cam_j.uid])
    match_pairs_name=[[pair[0].new_name, pair[1].new_name] for pair in match_pairs]
    return match_pairs_name, match_pairs


# def get_match_info(cam_1, cam_2):
#     return [cam_1.new_name, cam_2.new_name]

def get_match_info(cam_1, cam_2):
    return [cam_1, cam_2]


def pd_match(quadrant_pos, quadrant_dir, strict):
    match_list_v1 = [
        [7], [7, 8], [5, 6], [6], [3], [3, 4], [1, 2], [2]
    ]
    match_list_v2 = [
        [2, 3, 6, 7], [2, 3, 4, 6, 7, 8], [1, 2, 3, 5, 6, 7], [2, 3, 6, 7],
        [2, 3, 6, 7], [2, 3, 4, 6, 7, 8], [1, 2, 3, 5, 6, 7], [2, 3, 6, 7]
    ]
    match_list = []
    if strict:
        match_list = match_list_v1
    else:
        match_list = match_list_v2
    if quadrant_dir in match_list[quadrant_pos - 1]:
        return True
    else:
        return False


def dir_emb2quadrant(dir_emb):
    """
    :param dir_emb:     [dot,cross_x,cross_y]
    :return:
    """
    quadrant_list = [4, 1, 8, 5, 3, 2, 7, 6]
    return quadrant_list[dir_emb]


def pos_emb2quadrant(pos_emb):
    """
    :param pos_emb:   [x,y,z]
    :return:
    """
    quadrant_list = [8, 7, 4, 3, 5, 6, 1, 2]
    return quadrant_list[pos_emb]


def cal_embedding(pos_i, pos_j, dir_i, dir_j):
    delta_pos = pos_j - pos_i
    bi_vector = np.where(delta_pos > 0, 1, 0)
    bi_vector_str = np.array2string(bi_vector, separator='')[1:-1].replace(' ', '').replace('\n', '')

    position_embedding = int(bi_vector_str, 2)

    position_q = pos_emb2quadrant(position_embedding)

    dot_ans = np.dot(dir_i, dir_j)
    cross_ans = np.cross(dir_i, dir_j)

    bi_dot = np.where(dot_ans > 0, 1, 0)

    bi_cross = np.where(cross_ans > 0, 1, 0)

    bi_dot_str = str(bi_dot)

    bi_cross_str = np.array2string(bi_cross, separator='')[1:-1].replace(' ', '').replace('\n', '')[:2]
    dir_str = bi_dot_str + bi_cross_str

    direction_embedding = int(dir_str, 2)
    direction_q = dir_emb2quadrant(direction_embedding)

    return position_q, direction_q


### second step: concentric distance match
def concentric_match(cam_info_list, r, k, w):
    cam_num = len(cam_info_list)
    match_pairs = []
    match_pairs_C = []
    all_coords = cam_info_list[0].T
    for i in range(1, cam_num):
        all_coords = np.vstack([all_coords, cam_info_list[i].T])
    kdtree = cKDTree(all_coords)

    for c_ind, cam in enumerate(cam_info_list):
        query_point = cam.T[0]
        distances, indices = kdtree.query(query_point, k=cam_num)
        if c_ind>=1:
            match_pairs.append(get_match_info(cam, cam_info_list[c_ind-1]))
            match_pairs_C.append(get_match_info(cam, cam_info_list[c_ind-1]))
        for j in range(1, r + 1):
            match_pairs.append(get_match_info(cam, cam_info_list[indices[j]]))
        for j in range(r + 1, len(indices)):
            if (j - r) % k == 0:
                for jj in range(w):
                    if (jj + j) < len(indices):
                        match_pairs.append(get_match_info(cam, cam_info_list[indices[jj + j]]))
    match_pairs_name=[[pair[0].new_name, pair[1].new_name] for pair in match_pairs]

    return match_pairs_name, match_pairs, match_pairs_C


if __name__ == "__main__":
    remove_duplication(input_file='test_pairs.txt', output_file='test_pairs2.txt')
    # pass
    # fun_list = [{'name': "neighbor", 'fun': neighbor_n_pairs},
    #             {'name': "jump", 'fun': jump_k_pairs},
    #             {'name': "neighbor_jump", 'fun': neighbor_jump_nk_pairs}]
    #
    # folder_path = 'origin_images'
    # image_path = os.path.join(folder_path, 'input')
    # n = 5
    # k = 10
    # for fun in fun_list:
    #     fun_name = fun['name']
    #     function = fun['fun']
    #     if fun_name == 'neighbor':
    #         output_file = os.path.join(folder_path, fun_name + f'_{n}_pairs.txt')
    #     elif fun_name == 'jump':
    #         output_file = os.path.join(folder_path, fun_name + f'_{k}_pairs.txt')
    #     elif fun_name == 'neighbor_jump':
    #         output_file = os.path.join(folder_path, fun_name + f'_{n}_{k}_pairs.txt')
    #     function(image_path, n, k, output_file)
    # pos_1 = [0, 1, 0]
    # pos_2 = [2, 1, -1]
    # dir_1 = [1, 4.3, -2]
    # dir_2 = [1, 4.5, -1]
    # pos_1_np = np.array(pos_1)
    # pos_2_np = np.array(pos_2)
    # dir_1_np = np.array(dir_1)
    # dir_2_np = np.array(dir_2)
    # cal_embedding(pos_1_np, pos_2_np, dir_1_np, dir_2_np)
    # in_arr = np.array([1.2, 3, -4, 3.4, -0.4])
    # ret = np.where(in_arr > 0, 1, 0)
    # ret_str = np.array2string(ret, separator='')[1:-1].replace(' ', '').replace('\n', '')
    # print(ret)
    # print(ret_str)
