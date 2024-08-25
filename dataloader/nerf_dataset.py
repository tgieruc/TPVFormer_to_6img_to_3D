import os
import numpy as np
from mmcv.image.io import imread
import json
import random
from dataloader.transform_3d import NormalizeMultiviewImage
from torch.utils.data import DataLoader, Dataset
from dataloader.rays_dataset import RaysDataset
from kornia import create_meshgrid
from PIL import Image


class ClaraDataset(Dataset):
    """
    Dataset for 6img-to-3D. Each sample contains a NerfDataset
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config
        self.get_paths(dataset_config)


    def get_paths(self, dataset_config):
        """
        Aggregate the paths based on the dataset config
        """
        
        data_path = dataset_config.data_path
        self.data = []

        if dataset_config["town"] == "all":
            towns = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))]
        else:
            towns = dataset_config["town"]
        for town in towns:
            if dataset_config["weather"] == "all":
                weathers = [folder for folder in os.listdir(os.path.join(data_path, town)) if os.path.isdir(os.path.join(data_path, town, folder))]
            else:
                weathers = dataset_config["weather"]
            for weather in weathers:
                if dataset_config["vehicle"] == "all":
                    vehicles = [folder for folder in os.listdir(os.path.join(data_path, town, weather)) if os.path.isdir(os.path.join(data_path, town, weather, folder))]
                else:
                    vehicles = dataset_config["vehicle"]
                for vehicle in vehicles:
                    if dataset_config["spawn_point"] == ["all"]:
                        spawn_points = [folder for folder in os.listdir(os.path.join(data_path, town, weather, vehicle)) if "spawn_point_" in folder]
                    else:
                        spawn_points = [f"spawn_point_{i}" for i in dataset_config["spawn_point"]]
                    for spawn_point in spawn_points:
                        if dataset_config["step"] == ["all"]:
                            steps = [folder for folder in os.listdir(os.path.join(data_path, town, weather, vehicle, spawn_point)) if "step_" in folder]
                            steps = sorted(steps, key=lambda x: int(x.split('_')[1]))

                        else:
                            steps = [f"step_{i}" for i in dataset_config["step"]]
                        for step in steps:
                            self.data.append(os.path.join(data_path, town, weather, vehicle, spawn_point, step))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        return NeRFDataset(data, self.dataset_config)

    

class NeRFDataset(Dataset):
    """
    Dataset for 6img-to-3D. Each sample contains
    - input_views:
        - rgb:    n_input x H_input x W_input x 3
        - rays_o: n_input x H_input x W_input x 3
        - rays_d: n_input x H_input x W_input x 3
    - output_views:
        - rgb:   n_output x H_output x W_output x 3
        - rays_o: n_output x H_input x W_output x 3
        - rays_d: n_output x H_input x W_output x 3
    """

    def __init__(self, step_path, dataset_config):
        self.dataset_config = dataset_config
        self.step_path = step_path
        
        input_rgb, input_rays_o, input_rays_d = self.get_data(config=dataset_config.input_config)

    def get_data(self, config):
        transform_path = os.path.join(self.step_path, config.transform_path)

        with open(transform_path, 'r') as f:
            transforms = json.load(f)
        
        intrinsics_keys = ['w', 'h', 'fl_x', 'fl_y', 'cx', 'cy']

        default_intrinsics = {key: transforms.get(key, None) for key in intrinsics_keys}

        for frame in transforms['frames']:
            pose = np.array(frame['transform_matrix'])[:3, :4]
            intrinsics = {key: frame.get(key, default_intrinsics[key]) for key in intrinsics_keys}
            if config.scale != 1:
                intrinsics = scale_intrinsics(intrinsics, config.scale)
            direction = get_ray_directions(direction)
            rays_o, rays_d = get_rays(direction, pose)
            image_path = os.path.join(transform_path, f"{frame['file_path']}")
            img = Image.open(image_path)
            if config.scale != 1:
                img = img.resize((self.intrinsics.width, self.intrinsics.height), Image.LANCZOS)
            
        


def scale_intrinsics(intrinsics, factor: float):
    nw = round(intrinsics['w'] * factor)
    nh = round(intrinsics['w'] * factor)
    sw = nw / intrinsics['w']
    sh = nh / intrinsics['h']
    intrinsics['fl_x'] *= sw
    intrinsics['fl_y'] *= sh
    intrinsics['cx']*= sw
    intrinsics['cy']*= sh
    intrinsics['w'] = int(nw)
    intrinsics['h'] = int(nh)  

    return intrinsics  

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / np.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def get_ray_directions(intrinsics):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    H,W,focal = intrinsics['h'], intrinsics['w'], intrinsics['fl_x']
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        np.stack([(i-W/2)/focal, -(j-H/2)/focal, -np.ones_like(i)], -1) # (H, W, 3)

    return directions