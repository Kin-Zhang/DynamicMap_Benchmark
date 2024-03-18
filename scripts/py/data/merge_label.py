'''
# Created: 2024-03-18 21:14
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)

# Description:
# This script is used to convert point cloud data to semantic kitti format. since previously we saved inside pcd VIEWPOINT metadata

'''


import sys, os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)

import time, fire
from typing import List, Optional
import numpy as np
from tqdm import tqdm

from utils.pcdpy3 import save_pcd, load_pcd, xyzqwxyz_to_matrix
from utils import bc

# def toSemanticKITTI(
#     data_folder: str = "/home/kin/data/semindoor", # `pcd` folder inside this folder
# ):
#     # get the path of the data
#     pts_folder = os.path.join(data_folder, "pcd")
#     label_folder = os.path.join(data_folder, "labels")
#     print(f"pcd_files from folder: {pts_folder}")
#     # Check whether folder and files exist
#     assert os.path.isdir(pts_folder), f"{pts_folder} does not exist, {bc.FAIL}please check the path{bc.ENDC}"

#     # get the pcd files
#     pcd_files = sorted([os.path.join(pts_folder, i) for i in os.listdir(pts_folder) if i.endswith(".pcd")])
#     print(f"Total pcd files: {len(pcd_files)}")

#     # convert pcd to pose
#     poses = []
#     for i in pcd_files:
#         pcd_ = load_pcd(i)
#         poses.append(xyzqwxyz_to_matrix(list(pcd_.viewpoint)))
    
#     np.savetxt(fname=f"{data_folder}/poses.txt", X=np.array([np.concatenate((pose[0], pose[1], pose[2])) for pose in poses]))
#     print(f"Total poses: {len(poses)}, Save as KITTI format for point_labeler in: {data_folder}/poses.txt")
#     # TODO

def main(
    data_folder: str = "/home/kin/data/semindoor", # `pcd` folder inside this folder
):
    assert os.path.isdir(data_folder), f"{data_folder} does not exist, {bc.FAIL}please check the path{bc.ENDC}"
    # get the path of the data
    pts_folder = os.path.join(data_folder, "pcd")
    test_folder = os.path.join(data_folder, "test")
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    label_folder = os.path.join(data_folder, "labels")
    print(f"pcd_files from folder: {pts_folder}")

    # remove the .pcd extension
    file_names = sorted([i.split(".")[0] for i in os.listdir(pts_folder) if i.endswith(".pcd")])
    gt_maps = np.array([])

    for file_name in tqdm(file_names, desc="Merging label", ncols=80):
        pcd_ = load_pcd(os.path.join(pts_folder, file_name + ".pcd"))
        label_ = np.fromfile(os.path.join(label_folder, file_name + ".label"), dtype=np.uint32)

        sem_label = label_ & 0xFFFF
        # // transform intensity to label as uint32_t, check semantic KITTI config file:
        # // https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti-mos.yaml#L33-L41
        intensity = np.zeros_like(sem_label, dtype=np.float32)
        intensity[np.isin(sem_label, [252, 253, 254, 255, 256, 257, 258, 259])] = 1

        # save the pcd with label
        save_points = np.hstack((pcd_.np_data[:, :3], intensity.reshape(-1, 1))) # [x, y, z, intensity]
        gt_maps = np.vstack((gt_maps, save_points)) if gt_maps.size else save_points
        # save_pcd(save_pcd_file, save_points, np.array(list(pcd_.viewpoint)))
        save_pcd(os.path.join(test_folder, file_name + ".pcd"), save_points, np.array(list(pcd_.viewpoint)))
    
    save_pcd(os.path.join(data_folder, "gt_cloud.pcd"), gt_maps)
    print(f"The ground truth map is saved as: {data_folder}/gt_cloud.pcd. Now you have all evaluation data ready.")

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"Time used: {time.time() - start_time:.2f} s")