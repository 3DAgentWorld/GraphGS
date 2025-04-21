import networkx as nx
import json
import time

def read_graph(graph_path, cam_path):
    # Read camera information
    with open(cam_path, 'r') as json_file:
        camjson_data = json_file.read()
    camdic_list = json.loads(camjson_data)
    image_to_node_dic = {} 
    node_to_image_dic = {}
    for camdic in camdic_list:
        # Remove image extension
        image_name = camdic['image_name']
        if '.' in image_name:
            image_name = image_name.split('.')[0]
        
        image_to_node_dic[image_name] = int(camdic['uid'])
        node_to_image_dic[int(camdic['uid'])] = image_name

    # Read graph structure
    print("Loading graph structure...")
    start_time = time.time()
    with open(graph_path, 'r') as json_file:
        gjson_data = json_file.read()

    gdata = json.loads(gjson_data)
    # Convert node IDs to image names before building the graph
    # for link in gdata['links']:
    #     source_id = link['source']
    #     target_id = link['target']
    #     if source_id in node_to_image_dic:
    #         link['source_image'] = node_to_image_dic[source_id]
    #     if target_id in node_to_image_dic:
    #         link['target_image'] = node_to_image_dic[target_id]

    graph = nx.readwrite.json_graph.node_link_graph(gdata)
    
    # Store each node and its highest weight connection
    node_max_node_dic = {}
    for node_id in graph.nodes():
        if node_id in node_to_image_dic:
            source_name = node_to_image_dic[node_id]
            # Collect all edge weights
            edges_with_weights = []
            for _, target, data in graph.edges(node_id, data=True):
                if target in node_to_image_dic:
                    target_name = node_to_image_dic[target]
                    edges_with_weights.append((target, target_name, data['weight']))
            
            # Sort by weight
            edges_with_weights.sort(key=lambda x: x[2], reverse=True)
            
            # Store the fifth highest weight connection if it exists
            if len(edges_with_weights) >= 6:
                fifth_target = edges_with_weights[5][1]  # Get the node name of the fifth highest weight
                fifth_weight = edges_with_weights[5][2]  # Get the fifth highest weight value
                node_max_node_dic[source_name] = {
                    'target_node': fifth_target,
                    'weight': fifth_weight
                }
            else:
                # If there are fewer than 5 connections, store the last one or None
                if edges_with_weights:
                    last_target = edges_with_weights[-1][1]
                    last_weight = edges_with_weights[-1][2]
                    node_max_node_dic[source_name] = {
                        'target_node': last_target,
                        'weight': last_weight
                    }
                else:
                    node_max_node_dic[source_name] = {
                        'target_node': None,
                        'weight': 0.0
                    }
    
    
    # Calculate betweenness centrality, ignoring edge weights (all weights treated as 1)
    print("\nCalculating node betweenness centrality...")
    betweenness = nx.betweenness_centrality(graph, weight=None)

    # Normalize centrality weights
    max_centrality = max(betweenness.values()) if betweenness else 1.0
    min_centrality = min(betweenness.values()) if betweenness else 0.0
    
    # Check if all values are the same
    if max_centrality == min_centrality:
        normalized_betweenness = {node: 1.0 for node in betweenness}
    else:
        # Perform normalization (range from 0 to 1)
        normalized_betweenness = {
            node: (centrality - min_centrality) / (max_centrality - min_centrality)
            for node, centrality in betweenness.items()
        }
    
    # Calculate degree centrality
    print("\nCalculating node degree centrality...")
    degree_centrality = nx.degree_centrality(graph)
    
    # Normalize degree centrality
    max_degree_centrality = max(degree_centrality.values()) if degree_centrality else 1.0
    min_degree_centrality = min(degree_centrality.values()) if degree_centrality else 0.0
    
    # Check if all values are the same
    if max_degree_centrality == min_degree_centrality:
        normalized_degree_centrality = {node: 1.0 for node in degree_centrality}
    else:
        # Perform normalization (range from 0 to 1)
        normalized_degree_centrality = {
            node: (centrality - min_degree_centrality) / (max_degree_centrality - min_degree_centrality)
            for node, centrality in degree_centrality.items()
        }
    

    # Build node weight dictionary
    node_weight_dic = {}
    
    # Add betweenness centrality
    for node, centrality in normalized_betweenness.items():
        if node not in node_weight_dic:
            node_weight_dic[node] = {}
        node_weight_dic[node]['betweenness'] = centrality
    
    # Add degree centrality
    for node, centrality in normalized_degree_centrality.items():
        if node not in node_weight_dic:
            node_weight_dic[node] = {}
        node_weight_dic[node]['degree'] = centrality
        
    # Calculate combined weight (simple average)
    for node in node_weight_dic:
        betweenness_value = node_weight_dic[node].get('betweenness', 0.0)
        degree_value = node_weight_dic[node].get('degree', 0.0)
        node_weight_dic[node]['combined'] = (betweenness_value + degree_value) / 2

    # Build image name to weight mapping
    imagename_weight_dic = {}
    for camdic in camdic_list:
        node_id = int(camdic['uid'])
        if node_id in node_weight_dic:
            imagename_weight_dic[image_name] = node_weight_dic[node_id]
        else:
            print(f"Warning: Node ID {node_id} does not exist in the graph")
            imagename_weight_dic[camdic['image_name']] = 0.0
    return imagename_weight_dic, node_max_node_dic, graph

if __name__ == "__main__":
    graph_path='/home/chengchong/workspace/waymo/Q_concentric_5_10_1_loose/pairs_Q_concentric_5_10_1_loose_graphinfo.json'
    cam_path=graph_path.replace('graph','cam')
    image_to_node_dic, node_to_image_dic, node_weight_dic, imagename_weight_dic, graph=read_graph(graph_path, cam_path)
    image_name_demo='0_00000187'
    # if you want get node weight of image 0_00000187
    print(f'weigth of {image_name_demo}: {imagename_weight_dic[image_name_demo]}')
    # if you want get alledges of image node 0_00000187
    node_id=image_to_node_dic[image_name_demo]
    adjacent_edges = graph.edges(node_id, data=True)
    for edge in adjacent_edges:
        print(f"{edge[0]} - {edge[1]}, weight {edge[2]['weight']}")
        print(f"{node_to_image_dic[edge[0]]} - {node_to_image_dic[edge[1]]}, weight {edge[2]['weight']}")    