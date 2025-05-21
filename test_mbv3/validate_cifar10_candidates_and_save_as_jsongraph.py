import os
import re
import json

NUM_STAGE = 5
MAX_LAYER_PER_STAGE = 4
NUM_NODES = NUM_STAGE * MAX_LAYER_PER_STAGE
PADDING_TYPE_ID = 9

type_to_ks_e = {
    0: (3, 3), 1: (3, 4), 2: (3, 6),
    3: (5, 3), 4: (5, 4), 5: (5, 6),
    6: (7, 3), 7: (7, 4), 8: (7, 6),
}

# Linear reference adjacency matrix
adj_mat_ref = [[1 if abs(i - j) == 1 else 0 for j in range(NUM_NODES)] for i in range(NUM_NODES)]

def convert_nodes_to_ks_e_d(node_list):
    ks_list, e_list, d_list = [], [], []

    for i in range(NUM_STAGE):
        stage = node_list[i * MAX_LAYER_PER_STAGE : (i + 1) * MAX_LAYER_PER_STAGE]
        depth = 0
        for t in stage:
            if t == PADDING_TYPE_ID:
                break
            ks, e = type_to_ks_e.get(t, (-1, -1))
            ks_list.append(ks)
            e_list.append(e)
            depth += 1
        d_list.append(depth)

    return {'ks': ks_list, 'e': e_list, 'd': d_list}

def is_valid_mbv3(adj, ops):
    if not all(op in range(10) for op in ops[2:-1]):
        return False
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            if adj[i][j] not in [0, 1]:
                return False
            if j > i and adj[i][j] != adj_mat_ref[i][j]:
                return False
    return True

def main():
    input_path = "cifar10_final_graphs.txt"
    output_dir = "graphs"
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r") as f:
        text = f.read().replace("/n", "\n")

    graph_sections = text.strip().split("nodes:")[1:]

    valid_graphs = []
    for idx, section in enumerate(graph_sections):
        try:
            nodes_part, edges_part = section.split("edges:")
            nodes_raw = re.findall(r"tensor\((\d+)\)", nodes_part)
            edges_raw = re.findall(r"tensor\((\d+)\)", edges_part)

            if len(nodes_raw) != NUM_NODES or len(edges_raw) != NUM_NODES * NUM_NODES:
                print(f"Graph {idx} skipped: unexpected node/edge length.")
                continue

            nodes = list(map(int, nodes_raw))
            adj = [[int(edges_raw[i * NUM_NODES + j]) for j in range(NUM_NODES)] for i in range(NUM_NODES)]

            if not is_valid_mbv3(adj, nodes):
                print(f"Graph {idx} failed validation.")
                continue

            graph_data = {
                "id": idx,
                "nodes": nodes,
                "adj": adj,
                "ks_e_d": convert_nodes_to_ks_e_d(nodes)
            }
            valid_graphs.append(graph_data)

            out_path = os.path.join(output_dir, f"cifar10_graph_{len(valid_graphs)-1:03d}.json")
            with open(out_path, "w") as fout:
                json.dump(graph_data, fout, indent=2)

            print(f"Saved graph {idx} as {out_path}")

            if len(valid_graphs) == 100:
                break

        except Exception as e:
            print(f"Exception in graph {idx}: {e}")
            continue

    print(f"Total valid graphs saved: {len(valid_graphs)}")

if __name__ == "__main__":
    main()
