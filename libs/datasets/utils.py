"""
Code of "AV-Cloud: Spatial Audio Rendering Through Audio-Visual Cloud Splatting" 

Copyright (c) 2023-2024 University of Washington. 

Developed in UW NeuroAI Lab by Mingfei Chen (lasiafly@uw.edu).
"""


import os
import pickle

import numpy as np
from plyfile import PlyData
from sklearn.cluster import KMeans


def fetchPly(path, align_grids=None, N_points=256):
    if N_points is -1:
        points_path = path[:-len(os.path.basename(path))] + f'large_align_points_gs.pkl'
    else:
        points_path = path[:-len(os.path.basename(path))] + f'{N_points}_align_points_gs.pkl'
        
    if not os.path.exists(points_path):
        plydata = PlyData.read(path)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        points, colors, normals = get_grid(positions, colors, normals, align_grids, n_clusters=N_points, resolution=0.25)
        with open(points_path, 'wb') as file:
            points_dict = {
                'points': points,
                'colors': colors,
                'normals': normals
            }

            pickle.dump(points_dict, file)
    
    with open(points_path, 'rb') as file:
        points_dict = pickle.load(file)
    
    
    return points_dict['points'], points_dict['colors'], points_dict['normals']


def average_attributes(positions, attributes):
    """Average attributes (colors, normals) for unique positions."""
    unique_positions, indices = np.unique(positions, axis=0, return_inverse=True)

    # to avg
    averaged_attributes = np.array([attributes[indices == i].mean(axis=0) for i in range(len(unique_positions))])
    return unique_positions, averaged_attributes


def round_to_resolution(value, resolution):
    return np.round(value / resolution) * resolution



def vis(data):
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(data[:,0], data[:,1], data[:,2])

    # Label the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.title('3D Scatter Plot of Random Points')
    plt.savefig('3d.jpg')


def get_weighted_average(attributes, weights):
    """Calculate weighted average of attributes."""
    return np.average(attributes, axis=0, weights=weights)


def get_grid(positions, colors, normals, align_grids=None, n_clusters=256, resolution=0.25):
    k = 50
    rounded_positions = round_to_resolution(positions, resolution)
    # Average colors and normals for each unique rounded position
    unique_positions, averaged_colors = average_attributes(rounded_positions, colors)
    _, averaged_normals = average_attributes(rounded_positions, normals)
    
    positions = unique_positions
    colors = averaged_colors
    normals = averaged_normals

    if align_grids is not None:
        align_grids = align_grids.data.cpu().numpy()
    else:
        # Perform initial k-means clustering
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(positions)

        labels = kmeans.labels_
        align_grids = np.array(kmeans.cluster_centers_)
        
    closest_colors = np.zeros((align_grids.shape[0], k, 3))
    closest_normals = np.zeros((align_grids.shape[0], k, 3))

    for i, align_grid in enumerate(align_grids):
        distances = np.linalg.norm(unique_positions - align_grid, axis=1)
        closest_indices = np.argsort(distances)[:k]
        closest_colors[i] = colors[closest_indices]
        closest_normals[i] = normals[closest_indices]
    return align_grids, closest_colors.reshape(-1, 3*k), closest_normals.reshape(-1, 3*k)