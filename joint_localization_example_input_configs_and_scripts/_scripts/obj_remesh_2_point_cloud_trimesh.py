import argparse
import glob
import os

import numpy as np
import trimesh


def process_meshes(input_directory, output_directory, seed):
    for input_mesh_path in glob.glob(os.path.join(input_directory, '*.obj')):
        output_point_cloud_path = os.path.join(output_directory, os.path.splitext(os.path.basename(input_mesh_path))[0] + '.ply')
        process_mesh(input_mesh_path, output_point_cloud_path, seed)


def process_mesh_wrapper(args):
    input_mesh_path, output_directory, seed = args
    output_point_cloud_path = os.path.join(output_directory, os.path.splitext(os.path.basename(input_mesh_path))[0] + '.ply')
    process_mesh(input_mesh_path, output_point_cloud_path, seed)


def process_meshes_multiprocessing(input_directory, output_directory, seed):
    from multiprocessing import Pool
    with Pool() as pool:
        pool.map(process_mesh_wrapper, [(path, output_directory, seed) for path in glob.glob(os.path.join(input_directory, '*.obj'))])


def process_mesh(input_mesh_path, output_point_cloud_path=None, seed=0):
    print(f'Processing mesh: {input_mesh_path}')
          
    # Set output path if necessary
    if output_point_cloud_path is None:
        output_point_cloud_path = os.path.splitext(input_mesh_path)[0] + '.ply'

    # Set seed if necessary
    if seed is not None:
        np.random.seed(seed)
        
    # Loads either a mesh or a scene
    mesh_or_scene = trimesh.load_mesh(input_mesh_path)

    # Extract all meshes from the loaded scene if necessary
    if isinstance(mesh_or_scene, trimesh.Scene):
        meshes = [mesh_or_scene.geometry[key] for key in mesh_or_scene.geometry.keys()]
    else:
        meshes = [mesh_or_scene]

    # Merge all the meshes into a single mesh
    mesh = trimesh.util.concatenate(meshes)

    # Sample points on the mesh
    points = mesh.vertices
    # points, _ = trimesh.sample.sample_surface_even(mesh, count=12042) # TODO

    # Create a point cloud
    point_cloud = trimesh.PointCloud(points)

    # Create output directory if necessary
    os.makedirs(os.path.dirname(output_point_cloud_path), exist_ok=True)

    # Save the point cloud
    point_cloud.export(output_point_cloud_path)

    print(f'Finished processing mesh: {input_mesh_path}')


def main():
    parser = argparse.ArgumentParser(description='Convert meshes to point clouds')
    parser.add_argument('--input-directory', type=str, default='input', help='Input directory')
    parser.add_argument('--output-directory', type=str, default='output', help='Output directory')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    args = parser.parse_args()
    process_meshes_multiprocessing(args.input_directory, args.output_directory, args.seed)


if __name__ == '__main__':
   main()