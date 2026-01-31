"""
Convert JSON graph annotations to Trexplorer pickle format.

This script converts graph annotations in JSON format (with nodes, edges, and skeletons)
to the pickle format expected by Trexplorer Super.

Input format (JSON):
{
    "directed": true,
    "graph": {"coordinateSystem": "RAS"},
    "nodes": [{"id": 1, "pos": [x, y, z], "is_root": false}, ...],
    "edges": [{"source": 1, "target": 2, "skeletons": [[x,y,z], ...], "length": 10.5}, ...]
}

Output format (pickle):
{
    'branches': [bigtree.Node, ...],       # Tree structures
    'bifur_ids': [[node_ids], ...],        # Bifurcation node IDs per tree
    'endpts_ids': [[node_ids], ...],       # Endpoint node IDs per tree
    'root_id': [node_ids],                 # Root node IDs
    'interm_ids': [[node_ids], ...],       # Intermediate node IDs per tree
    'branch_ids': [[branch_ids], ...],     # Branch IDs per tree
    'all_ids': [[node_ids], ...],          # All node IDs per tree
    'num_points': [[counts], ...],         # Points per branch per tree
    'networkx': [nx.DiGraph, ...],         # NetworkX graphs
    'trajectories': [[{path, bifur_ids, endpt_id}], ...]  # Paths from root to endpoints
}

Usage:
    python convert_json_to_trexplorer.py --input_dir /path/to/json_files \
                                         --mask_dir /path/to/masks \
                                         --output_dir /path/to/output \
                                         --spacing 0.5
"""

import os
import json
import pickle
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from scipy.ndimage import distance_transform_edt

os.environ["BIGTREE_CONF_ASSERTIONS"] = ""
from bigtree import Node

try:
    import nibabel as nib
except ImportError:
    print("Warning: nibabel not installed. Radius estimation from masks will not work.")
    nib = None


def load_json_graph(json_path: str) -> dict:
    """Load JSON graph annotation."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_mask_and_compute_distance_transform(mask_path: str) -> np.ndarray:
    """Load segmentation mask and compute distance transform for radius estimation."""
    if nib is None:
        raise ImportError("nibabel is required for mask loading")

    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_fdata().astype(np.uint8)

    # Distance transform gives distance to nearest background voxel
    distance_map = distance_transform_edt(mask > 0)

    return distance_map


def get_radius_at_position(pos: list, distance_map: np.ndarray, spacing: float = 1.0) -> float:
    """Get radius at a position from the distance transform."""
    # Round to nearest voxel
    x, y, z = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))

    # Clamp to valid range
    x = max(0, min(x, distance_map.shape[0] - 1))
    y = max(0, min(y, distance_map.shape[1] - 1))
    z = max(0, min(z, distance_map.shape[2] - 1))

    # Return radius in mm (distance_map is in voxels)
    return float(distance_map[x, y, z]) * spacing


def build_networkx_graph(json_data: dict) -> nx.DiGraph:
    """Build a NetworkX directed graph from JSON data."""
    G = nx.DiGraph()

    # Add nodes
    for node in json_data['nodes']:
        G.add_node(node['id'], pos=node['pos'], is_root=node.get('is_root', False))

    # Add edges with skeleton points
    for edge in json_data['edges']:
        G.add_edge(edge['source'], edge['target'],
                   skeletons=edge.get('skeletons', []),
                   length=edge.get('length', 0))

    return G


def find_root_nodes(G: nx.DiGraph) -> list:
    """Find root nodes (nodes marked as is_root or with no incoming edges)."""
    roots = []
    for node in G.nodes():
        if G.nodes[node].get('is_root', False):
            roots.append(node)

    # If no nodes marked as root, use nodes with no predecessors
    if not roots:
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]

    return roots


def compute_node_degrees(G: nx.DiGraph) -> dict:
    """Compute total degree (in + out) for each node."""
    degrees = {}
    for node in G.nodes():
        degrees[node] = G.in_degree(node) + G.out_degree(node)
    return degrees


def classify_node(G: nx.DiGraph, node_id: int) -> int:
    """
    Classify node type:
    - 0: endpoint (degree 1, leaf node)
    - 1: intermediate (degree 2, on a path)
    - 2: bifurcation (degree > 2, branching point)
    """
    in_deg = G.in_degree(node_id)
    out_deg = G.out_degree(node_id)
    total_deg = in_deg + out_deg

    if out_deg == 0:  # No children = endpoint
        return 0
    elif out_deg == 1:  # One child = intermediate
        return 1
    else:  # Multiple children = bifurcation
        return 2


def expand_edge_to_nodes(edge_data: dict, source_pos: list, target_pos: list,
                         distance_map: np.ndarray = None, spacing: float = 1.0) -> list:
    """
    Expand an edge with skeleton points into a list of intermediate positions.
    Returns list of (position, radius) tuples.
    """
    skeletons = edge_data.get('skeletons', [])

    if len(skeletons) < 2:
        # No intermediate points, just return source and target
        return []

    # Skeletons include source and target, we want intermediate points only
    intermediate_points = skeletons[1:-1]  # Exclude first and last (source/target)

    result = []
    for pos in intermediate_points:
        if distance_map is not None:
            radius = get_radius_at_position(pos, distance_map, spacing)
        else:
            radius = 1.0  # Default radius
        result.append((pos, radius))

    return result


def build_tree_from_root(G: nx.DiGraph, root_id: int,
                         distance_map: np.ndarray = None,
                         spacing: float = 1.0) -> tuple:
    """
    Build a bigtree Node structure from a NetworkX graph starting at root.
    Also expands edge skeletons into intermediate nodes.

    Returns:
        - root_node: bigtree.Node root
        - all_ids: list of all node IDs in order
        - branch_ids: list of branch IDs
        - num_points: list of point counts per branch
        - bifur_ids: list of bifurcation node IDs
        - endpts_ids: list of endpoint node IDs
        - interm_ids: list of intermediate node IDs
        - nx_graph: NetworkX representation of the expanded tree
    """
    # Track branch assignments
    branch_counter = [0]  # Use list to allow modification in nested function

    # Storage for node info
    all_node_ids = []
    branch_ids_set = set()
    points_per_branch = defaultdict(int)
    bifur_ids = []
    endpts_ids = []
    interm_ids = []

    # Create NetworkX graph for the expanded tree
    nx_tree = nx.DiGraph()

    def get_node_radius(pos):
        if distance_map is not None:
            return get_radius_at_position(pos, distance_map, spacing)
        return 1.0

    def create_node_name(branch_id: int, point_id: int) -> str:
        return f"{branch_id}-{point_id}"

    def process_node(nx_node_id: int, parent_bigtree: Node, current_branch: int,
                     point_in_branch: int, is_new_branch: bool) -> Node:
        """Recursively process a node and its children."""

        pos = G.nodes[nx_node_id]['pos']
        radius = get_node_radius(pos)
        label = classify_node(G, nx_node_id)

        node_name = create_node_name(current_branch, point_in_branch)

        # Create bigtree node
        if parent_bigtree is None:
            bt_node = Node(node_name, position=pos, radius=radius, label=label)
        else:
            bt_node = Node(node_name, position=pos, radius=radius, label=label,
                          parent=parent_bigtree)

        # Track in NetworkX
        nx_tree.add_node(node_name, position=pos, radius=radius, label=label,
                        original_id=nx_node_id)
        if parent_bigtree is not None:
            nx_tree.add_edge(parent_bigtree.name, node_name)

        # Track IDs
        all_node_ids.append(node_name)
        branch_ids_set.add(str(current_branch))
        points_per_branch[current_branch] += 1

        # Classify
        if label == 0:
            endpts_ids.append(node_name)
        elif label == 2:
            bifur_ids.append(node_name)
        else:
            interm_ids.append(node_name)

        # Process children
        children = list(G.successors(nx_node_id))

        for i, child_id in enumerate(children):
            # Get edge data
            edge_data = G.edges[nx_node_id, child_id]

            # Expand skeleton points
            child_pos = G.nodes[child_id]['pos']
            intermediate_points = expand_edge_to_nodes(edge_data, pos, child_pos,
                                                       distance_map, spacing)

            # Determine branch for this child
            if len(children) > 1:
                # Bifurcation: each child starts a new branch
                branch_counter[0] += 1
                child_branch = branch_counter[0]
                child_point_id = 0
            else:
                # Continue same branch
                child_branch = current_branch
                child_point_id = point_in_branch + 1

            # Add intermediate points
            current_parent = bt_node
            current_point_id = child_point_id
            for inter_pos, inter_radius in intermediate_points:
                inter_name = create_node_name(child_branch, current_point_id)
                inter_node = Node(inter_name, position=inter_pos, radius=inter_radius,
                                 label=1, parent=current_parent)

                # Track in NetworkX
                nx_tree.add_node(inter_name, position=inter_pos, radius=inter_radius, label=1)
                nx_tree.add_edge(current_parent.name, inter_name)

                # Track IDs
                all_node_ids.append(inter_name)
                branch_ids_set.add(str(child_branch))
                points_per_branch[child_branch] += 1
                interm_ids.append(inter_name)

                current_parent = inter_node
                current_point_id += 1

            # Process child node
            process_node(child_id, current_parent, child_branch, current_point_id,
                        len(children) > 1)

        return bt_node

    # Start building from root
    root_pos = G.nodes[root_id]['pos']
    root_node = process_node(root_id, None, 0, 0, True)

    # Compile results
    branch_ids = sorted(list(branch_ids_set), key=int)
    num_points = [points_per_branch[int(b)] for b in branch_ids]

    return (root_node, all_node_ids, branch_ids, num_points,
            bifur_ids, endpts_ids, interm_ids, nx_tree)


def compute_trajectories(root_node: Node, all_ids: list, bifur_ids: list,
                         endpts_ids: list) -> list:
    """
    Compute trajectories from root to each endpoint.
    Each trajectory is a dict with:
    - path: list of node IDs from root to endpoint
    - bifur_ids: bifurcation points along this path
    - endpt_id: the endpoint ID
    """
    from bigtree import find_name

    trajectories = []
    bifur_set = set(bifur_ids)

    for endpt_id in endpts_ids:
        # Find the endpoint node
        endpt_node = find_name(root_node, endpt_id)
        if endpt_node is None:
            continue

        # Build path from root to endpoint
        path = []
        current = endpt_node
        while current is not None:
            path.append(current.name)
            current = current.parent
        path = path[::-1]  # Reverse to go from root to endpoint

        # Find bifurcations along this path
        path_bifurs = [nid for nid in path if nid in bifur_set]

        trajectories.append({
            'path': path,
            'bifur_ids': path_bifurs,
            'endpt_id': endpt_id
        })

    return trajectories


def convert_single_sample(json_path: str, mask_path: str = None,
                          spacing: float = 1.0, default_radius: float = 1.0) -> dict:
    """
    Convert a single JSON graph to Trexplorer pickle format.

    Args:
        json_path: Path to JSON graph file
        mask_path: Path to segmentation mask (.nii.gz) for radius estimation
        spacing: Voxel spacing in mm
        default_radius: Default radius if no mask provided

    Returns:
        Dictionary in Trexplorer pickle format
    """
    # Load JSON
    json_data = load_json_graph(json_path)

    # Load distance map if mask provided
    distance_map = None
    if mask_path and os.path.exists(mask_path):
        distance_map = load_mask_and_compute_distance_transform(mask_path)

    # Build NetworkX graph
    G = build_networkx_graph(json_data)

    # Find roots
    roots = find_root_nodes(G)
    if not roots:
        raise ValueError(f"No root nodes found in {json_path}")

    # Build trees for each root (handles disconnected components)
    all_branches = []
    all_bifur_ids = []
    all_endpts_ids = []
    all_interm_ids = []
    all_branch_ids = []
    all_num_points = []
    all_ids = []
    all_nx_graphs = []
    all_trajectories = []
    root_ids = []

    for root_id in roots:
        # Get connected component for this root
        (root_node, node_ids, branch_ids, num_points,
         bifur_ids, endpts_ids, interm_ids, nx_graph) = build_tree_from_root(
            G, root_id, distance_map, spacing
        )

        # Compute trajectories
        trajectories = compute_trajectories(root_node, node_ids, bifur_ids, endpts_ids)

        all_branches.append(root_node)
        all_bifur_ids.append(bifur_ids)
        all_endpts_ids.append(endpts_ids)
        all_interm_ids.append(interm_ids)
        all_branch_ids.append(branch_ids)
        all_num_points.append(num_points)
        all_ids.append(node_ids)
        all_nx_graphs.append(nx_graph)
        all_trajectories.append(trajectories)
        root_ids.append(root_node.name)

    # Build output dictionary
    result = {
        'branches': all_branches,
        'bifur_ids': all_bifur_ids,
        'endpts_ids': all_endpts_ids,
        'root_id': root_ids,
        'interm_ids': all_interm_ids,
        'branch_ids': all_branch_ids,
        'all_ids': all_ids,
        'num_points': all_num_points,
        'networkx': all_nx_graphs,
        'trajectories': all_trajectories
    }

    return result


def convert_dataset(input_dir: str, mask_dir: str, output_dir: str,
                    spacing: float = 1.0, json_suffix: str = '.graph.json',
                    mask_suffix: str = '.label.nii.gz'):
    """
    Convert an entire dataset of JSON graphs to Trexplorer pickle format.

    Args:
        input_dir: Directory containing JSON graph files
        mask_dir: Directory containing segmentation masks
        output_dir: Directory to save pickle files
        spacing: Voxel spacing in mm
        json_suffix: Suffix for JSON files
        mask_suffix: Suffix for mask files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all JSON files
    json_files = list(Path(input_dir).glob(f'*{json_suffix}'))

    if not json_files:
        print(f"No JSON files found with suffix '{json_suffix}' in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files to convert")

    for json_path in tqdm(json_files, desc="Converting"):
        # Determine sample ID from filename
        sample_name = json_path.name.replace(json_suffix, '')

        # Find corresponding mask
        mask_path = None
        if mask_dir:
            mask_path = os.path.join(mask_dir, sample_name + mask_suffix)
            if not os.path.exists(mask_path):
                # Try alternative naming
                mask_path = os.path.join(mask_dir, sample_name + '.nii.gz')
                if not os.path.exists(mask_path):
                    print(f"Warning: No mask found for {sample_name}, using default radius")
                    mask_path = None

        try:
            result = convert_single_sample(str(json_path), mask_path, spacing)

            # Save pickle
            output_path = os.path.join(output_dir, f"{sample_name}.pickle")
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)

        except Exception as e:
            print(f"Error converting {json_path}: {e}")
            continue

    print(f"Conversion complete. Output saved to {output_dir}")


def organize_converted_dataset(data_dir: str, output_dir: str,
                               train_ratio: float = 0.7,
                               val_ratio: float = 0.15,
                               test_ratio: float = 0.15,
                               seed: int = 42,
                               annot_dir: str = None):
    """
    Organize converted dataset into Trexplorer directory structure.

    Args:
        data_dir: Directory containing images and masks
        output_dir: Destination directory
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
        annot_dir: Directory containing annotations (default: data_dir/annotations)
    """
    import shutil
    import random

    random.seed(seed)

    # Create output directories
    subdirs = [
        'annots_train', 'annots_val', 'annots_val_sub_vol', 'annots_test',
        'images_train', 'images_val', 'images_val_sub_vol', 'images_test',
        'masks_train', 'masks_val', 'masks_val_sub_vol', 'masks_test'
    ]

    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    # Find all annotation files
    if annot_dir is None:
        annot_dir = os.path.join(data_dir, 'annotations')
    annot_files = sorted(Path(annot_dir).glob('*.pickle'))

    if not annot_files:
        print(f"No pickle files found in {annot_dir}")
        return

    # Shuffle and split
    samples = [f.stem for f in annot_files]
    random.shuffle(samples)

    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    print(f"Split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")

    def copy_sample(sample_name: str, split: str, idx: int):
        """Copy a sample's files to the appropriate split directories."""
        # Find source files (handle different naming conventions and directory structures)
        img_patterns = [f'{sample_name}.img.nii.gz', f'{sample_name}.nii.gz']
        mask_patterns = [f'{sample_name}.label.nii.gz', f'{sample_name}.nii.gz']

        # Search in both data_dir/images/ and data_dir/ directly
        img_dirs = [os.path.join(data_dir, 'images'), data_dir]
        mask_dirs = [os.path.join(data_dir, 'masks'), data_dir]

        img_src = None
        for img_dir in img_dirs:
            for pattern in img_patterns:
                path = os.path.join(img_dir, pattern)
                if os.path.exists(path):
                    img_src = path
                    break
            if img_src:
                break

        mask_src = None
        for mask_dir in mask_dirs:
            for pattern in mask_patterns:
                path = os.path.join(mask_dir, pattern)
                if os.path.exists(path):
                    mask_src = path
                    break
            if mask_src:
                break

        annot_src = os.path.join(annot_dir, f'{sample_name}.pickle')

        # Copy to destination
        dst_name = f'{idx}.nii.gz'
        dst_annot = f'{idx}.pickle'

        if img_src:
            shutil.copy(img_src, os.path.join(output_dir, f'images_{split}', dst_name))
            if split == 'val':
                shutil.copy(img_src, os.path.join(output_dir, 'images_val_sub_vol', dst_name))

        if mask_src:
            shutil.copy(mask_src, os.path.join(output_dir, f'masks_{split}', dst_name))
            if split == 'val':
                shutil.copy(mask_src, os.path.join(output_dir, 'masks_val_sub_vol', dst_name))

        if os.path.exists(annot_src):
            shutil.copy(annot_src, os.path.join(output_dir, f'annots_{split}', dst_annot))
            if split == 'val':
                shutil.copy(annot_src, os.path.join(output_dir, 'annots_val_sub_vol', dst_annot))

    # Process each split
    for idx, sample in enumerate(tqdm(train_samples, desc="Copying train")):
        copy_sample(sample, 'train', idx)

    for idx, sample in enumerate(tqdm(val_samples, desc="Copying val")):
        copy_sample(sample, 'val', idx)

    for idx, sample in enumerate(tqdm(test_samples, desc="Copying test")):
        copy_sample(sample, 'test', idx)

    print(f"Dataset organized in {output_dir}")
    print("Don't forget to run generate_val_sub_vol_file.py to create annots_val_sub_vol.pickle")


def main():
    parser = argparse.ArgumentParser(
        description='Convert JSON graph annotations to Trexplorer pickle format'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert JSON files to pickle')
    convert_parser.add_argument('--input_dir', required=True,
                               help='Directory containing JSON graph files')
    convert_parser.add_argument('--mask_dir', default=None,
                               help='Directory containing segmentation masks')
    convert_parser.add_argument('--output_dir', required=True,
                               help='Directory to save pickle files')
    convert_parser.add_argument('--spacing', type=float, default=1.0,
                               help='Voxel spacing in mm (default: 1.0)')
    convert_parser.add_argument('--json_suffix', default='.graph.json',
                               help='Suffix for JSON files (default: .graph.json)')
    convert_parser.add_argument('--mask_suffix', default='.label.nii.gz',
                               help='Suffix for mask files (default: .label.nii.gz)')

    # Organize command
    organize_parser = subparsers.add_parser('organize',
                                            help='Organize into Trexplorer directory structure')
    organize_parser.add_argument('--data_dir', required=True,
                                help='Directory with images and masks')
    organize_parser.add_argument('--annot_dir', default=None,
                                help='Directory with annotations (default: data_dir/annotations)')
    organize_parser.add_argument('--output_dir', required=True,
                                help='Destination directory')
    organize_parser.add_argument('--train_ratio', type=float, default=0.7)
    organize_parser.add_argument('--val_ratio', type=float, default=0.15)
    organize_parser.add_argument('--test_ratio', type=float, default=0.15)
    organize_parser.add_argument('--seed', type=int, default=42)

    # Single file command
    single_parser = subparsers.add_parser('single', help='Convert a single file')
    single_parser.add_argument('--json', required=True, help='JSON file path')
    single_parser.add_argument('--mask', default=None, help='Mask file path')
    single_parser.add_argument('--output', required=True, help='Output pickle path')
    single_parser.add_argument('--spacing', type=float, default=1.0)

    args = parser.parse_args()

    if args.command == 'convert':
        convert_dataset(args.input_dir, args.mask_dir, args.output_dir,
                       args.spacing, args.json_suffix, args.mask_suffix)

    elif args.command == 'organize':
        organize_converted_dataset(args.data_dir, args.output_dir,
                                  args.train_ratio, args.val_ratio,
                                  args.test_ratio, args.seed,
                                  args.annot_dir)

    elif args.command == 'single':
        result = convert_single_sample(args.json, args.mask, args.spacing)
        with open(args.output, 'wb') as f:
            pickle.dump(result, f)
        print(f"Saved to {args.output}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
