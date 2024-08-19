import open3d as o3d
import torch
import kaolin
import numpy as np
import os
import argparse

def load_point_cloud(file_path, sample_points=100000):
    """
    Load a point cloud or 3D mesh from a .obj or .ply file and convert its vertices to a Tensor.
    """
    file_extension = file_path.split('.')[-1]
    points = None
    
    if file_extension.lower() == 'ply':
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
    elif file_extension.lower() == 'obj':
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertices():
            raise ValueError("The OBJ file does not contain any vertices.")
        points = np.asarray(mesh.vertices)
    else:
        raise ValueError("Unsupported file format. Only .obj and .ply files are supported.")
    
    
    if len(points) > sample_points:
        # random sampling
        # indices = np.random.choice(len(points), sample_points, replace=False)
        # points = points[indices, :]
        points = uniform_sampling(points, sample_points)
        
    elif len(points) < sample_points:
        # If the number of points is less than the desired number of samples, you can choose to either resample with repetition or retain the existing point cloud.
        points = np.pad(points, ((0, max(0, sample_points - len(points))), (0, 0)), mode='wrap')
        # points = points
    
    # Convert a numpy array to a torch Tensor
    # points_tensor = torch.tensor(points, dtype=torch.float).unsqueeze(0)  
    return points
def uniform_sampling(points, num_samples):
    """
    Uniformly sample by creating a cubic grid over the point cloud and selecting points from each non-empty grid cell.

    Parameters:

    points: Numpy array of the original point cloud.
    num_samples: Desired number of samples.
    Returns:

    sampled_points: The sampled point cloud.
    """
    # Calculate the bounding box.
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    # Calculate the grid size.
    num_cells = int(np.ceil(num_samples ** (1. / 3)))
    grid_size = (max_bound - min_bound) / num_cells
    grid = {}

    # Place each point into the corresponding grid cell
    for point in points:
        grid_index = np.floor((point - min_bound) / grid_size).astype(int)
        grid_index = tuple(grid_index)
        if grid_index not in grid:
            grid[grid_index] = []
        grid[grid_index].append(point)
    
    # Sample one point from each non-empty grid cell.
    sampled_points = []
    while len(sampled_points) < num_samples and len(grid) > 0:
        grid_index = list(grid.keys())[np.random.randint(len(grid))]
        sampled_points.append(grid[grid_index].pop(np.random.randint(len(grid[grid_index]))))
        if len(grid[grid_index]) == 0:
            del grid[grid_index]

    return np.array(sampled_points)
def icp_align(source_points, target_points, threshold=1):
    """
    Align two point clouds using the ICP algorithm.

    Parameters:

    source_points: Numpy array of the source point cloud with shape (N, 3).
    target_points: Numpy array of the target point cloud with shape (M, 3).
    threshold: Distance threshold for the ICP algorithm.
    Returns:

    aligned_source_points: Numpy array of the aligned source point cloud with shape (N, 3).
    """

    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_points)
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_points)
    
    # ICP
    icp_result = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 2000)
        )
    
    # Obtain the transformation matrix and apply it to the source point cloud.
    transformation = icp_result.transformation
    target_cloud.transform(transformation)
    
    # Convert the aligned point cloud back to a numpy array.
    aligned_target_points = np.asarray(target_cloud.points)

    # source_tensor = torch.tensor(source_points, dtype=torch.float).unsqueeze(0)
    # aligned_target_points_tensor = torch.tensor(aligned_target_points, dtype=torch.float).unsqueeze(0)  
    
    return source_points,aligned_target_points
def scale_normalize(source_points, points):
    source_min = np.min(source_points, axis=0)
    source_max = np.max(source_points, axis=0)
    source_size = np.max(source_max - source_min)
    
    points_min = np.min(points, axis=0)
    points_max = np.max(points, axis=0)
    points_size = np.max(points_max - points_min)
    
    scale_factor = source_size / points_size
    points_center = (points_max + points_min) / 2
    normalized_points = (points - points_center) * scale_factor + points_center
    
    return normalized_points

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gt-path",
    default="",
    type=str,
    help="ground truth path to save the obj, ply file",
)
parser.add_argument(
    "--output-path",
    default="",
    type=str,
    help="output path to save the obj, ply file",
)

args = parser.parse_args()

gt_path = args.gt_path
output_path = args.output_path

gt = load_point_cloud(gt_path)
output = load_point_cloud(output_path) 

source_tensor = torch.tensor(gt, dtype=torch.float).unsqueeze(0)
target_tensor = torch.tensor(output, dtype=torch.float).unsqueeze(0)  

if torch.cuda.is_available():
    source_tensor = source_tensor.to('cuda')
    target_tensor = target_tensor.to('cuda')
else:
    print("CUDA is not available. Please ensure that your environment is correctly configured and that your GPU supports CUDA.")

cd = kaolin.metrics.pointcloud.chamfer_distance(source_tensor, target_tensor)
f2 = kaolin.metrics.pointcloud.f_score(source_tensor, target_tensor,radius=0.2) 
f3 = kaolin.metrics.pointcloud.f_score(source_tensor, target_tensor,radius=0.3) 
f4 = kaolin.metrics.pointcloud.f_score(source_tensor, target_tensor,radius=0.4) 

print(f"CD:{str(cd.item()).encode('utf-8').decode('utf-8')}    F score(0.2):{str(f2.item()).encode('utf-8').decode('utf-8')}    F score(0.3):{str(f3.item()).encode('utf-8').decode('utf-8')}    Fscore(0.4):{str(f4.item()).encode('utf-8').decode('utf-8')}")
