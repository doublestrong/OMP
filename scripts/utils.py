import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as sciR


def quat2mat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    # check quaternion
    assert abs(qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2 - 1.0) < 1e-4
    return np.array([[1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                     [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qz * qy - 2 * qx * qw],
                     [2 * qx * qz - 2 * qy * qw, 2 * qz * qy + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]])


class Pose3:
    def __init__(self, mat: np.ndarray):
        self.mat = mat

    @classmethod
    def by_vec(cls, x: float, y: float, z: float, qw: float, qx: float, qy: float, qz: float):
        # convert to quaternion
        mat = np.eye(4)
        mat[:3, :3] = quat2mat(qw, qx, qy, qz)
        mat[:3, 3] = np.array([x, y, z])
        return cls(mat)

    @classmethod
    def from2d(cls, x: float, y: float, theta: float):
        # convert to quaternion
        mat = np.eye(4)
        mat[:3, 3] = np.array([x, y, .0])
        mat[:3, 0] = [np.cos(theta), np.sin(theta), .0]
        mat[:3, 1] = [-np.sin(theta), np.cos(theta), .0]
        return cls(mat)


    @property
    def translation(self):
        return self.mat[:3, 3]

    @property
    def rotation(self):
        return self.mat[:3, :3]

    @property
    def transQuat(self):
        """
        :return: tx ty tz qw qx qy qz
        """
        tmp = sciR.from_matrix(self.rotation).as_quat() # xyzw
        tmp2 = np.zeros(4)
        tmp2[0] = tmp[-1]
        tmp2[1:] = tmp[:3]
        return np.concatenate((self.translation,tmp2))

    def inverse(self):
        mat = self.mat.copy()
        mat[:3, :3] = mat[:3, :3].T
        mat[:3, 3] = -mat[:3, :3] @ mat[:3, 3]
        return Pose3(mat)

    def tf_point(self, arr: np.array):
        vec = np.array([arr[0], arr[1], arr[2], 1])
        return (self.mat @ vec)[:3]

    def tf_points(self, points: np.array):
        h_points = np.hstack([points, np.ones((len(points), 1))])
        return (self.mat @ h_points.T).T[:, :3]

    def __mul__(self, other):
        if isinstance(other, Pose3):
            return Pose3(self.mat @ other.mat)
        raise ValueError("Not a Pose3 type to multiply.")

def read_pfm(filename):
    # https: // stackoverflow.com / questions / 48809433 / read - pfm - format - in -python
    with Path(filename).open('rb') as pfm_file:
        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')

        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4

        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.reshape(decoded, shape) * scale


def px2coor(w, h, px_y: np.ndarray, px_z: np.ndarray, px_d: np.ndarray, focal_len, dep_type="Planar"):
    """
    :param w: image width
    :param w: image height
    :param px_y: horizontal coordinates of pixels in the image
    :param px_z: vertical coordinates of pixels in the image
    :param px_d: perspective or planar depth
    :param focal_len: focal length in pixels
    :return: ndarray of x, y, z coordinates in the NED camera frame
    """
    c_h, c_w = h / 2.0, w / 2.0
    new_y = px_y + .5 - c_w
    new_z = px_z + .5 - c_h
    if dep_type == "Planar":
        real2px = px_d / focal_len
        real_x = px_d
        real_y = real2px * new_y
        real_z = real2px * new_z
    elif dep_type == "Perspective":
        new_d = np.sqrt(focal_len ** 2 + new_z ** 2 + new_y ** 2)
        real2px = px_d / new_d
        real_x = real2px * focal_len
        real_y = real2px * new_y
        real_z = real2px * new_z
    else:
        raise ValueError("Unknown dep image type.")
    return np.hstack((real_x.reshape((-1, 1)),
                      real_y.reshape((-1, 1)),
                      real_z.reshape((-1, 1))))

def depth_interpolation(dep_img: np.ndarray, rgb_pixels: np.ndarray, dep_img_scale: float):
    scaled_pix = rgb_pixels * dep_img_scale
    scaled_pix = scaled_pix.astype(int)
    return dep_img[tuple(scaled_pix.T)]

def ray_bearing_width(w, focal_len):
    """
    :param w: width of image, int
    :param focal_len: focal length in pixels
    :return: bearing and width of a beam in ray casting; bearing on the left is positive
    """
    c_w = w / 2.0
    ray_y_arr = -np.arange(0, w, 1) + c_w - .5
    ray_bearing = np.arctan2(ray_y_arr, focal_len)
    ray_l_edge_arr = -np.arange(0, w, 1) + c_w
    ray_l_edge_bearing = np.arctan2(ray_l_edge_arr, focal_len)
    ray_r_edge_bearing = ray_l_edge_bearing.copy()
    ray_r_edge_bearing[-1] = -ray_l_edge_bearing[0]
    ray_r_edge_bearing[:-1] = ray_l_edge_bearing[1:]
    return ray_bearing, abs(ray_r_edge_bearing - ray_l_edge_bearing)


def Pose3OnXY(pose: Pose3):
    return pose.mat[0, 3], pose.mat[1, 3], np.arctan2(pose.mat[1, 0], pose.mat[0, 0])


@dataclass
class PoseImgData:
    TimeStamp: int
    Pose: Pose3
    RGBPath: str
    SegPath: str
    DepPath: str
    Normals: str


class ParamStruct:
    def __init__(self, yaml_setting):
        self.random_seed = yaml_setting['random_seed']
        self.setting_dict = yaml_setting
        self.data_dir = yaml_setting['data_dir']
        self.cam_name = str(yaml_setting['airsim']['cam_name'])
        self.wall_labels = yaml_setting['airsim']['wall_labels']
        self.cube_label = yaml_setting['airsim']['cube_label']
        self.fine_angular_resolution = yaml_setting['angular_scan_resolution']['fine']
        self.coarse_angular_resolution = yaml_setting['angular_scan_resolution']['coarse']
        self.img_w = yaml_setting['airsim']['img_w']
        self.img_h = yaml_setting['airsim']['img_h']
        self.dep_w = yaml_setting['airsim']['dep_w']
        self.dep_h = yaml_setting['airsim']['dep_h']
        self.fov = yaml_setting['airsim']['fov']
        self.grid_size = yaml_setting['grid_mapping']['grid_size']
        self.grid_x_len = yaml_setting['grid_mapping']['x_length']
        self.grid_y_len = yaml_setting['grid_mapping']['y_length']
        self.grid_x_offset = yaml_setting['grid_mapping']['x_offset']
        self.grid_y_offset = yaml_setting['grid_mapping']['y_offset']
        self.depth_max = yaml_setting['depth_max']
        self.obstacle_thickness = yaml_setting['grid_mapping']['obstacle_thickness']
        self.free_dot_threshold = yaml_setting['grid_mapping']['free_dot_threshold']
        self.manual_align = yaml_setting['grid_mapping']['manual_align']
        self.plot_depth = yaml_setting['plot_depth']
        self.free_bearing_inflate = yaml_setting['grid_mapping']['free_bearing_inflate']
        self.ocp_bearing_inflate = yaml_setting['grid_mapping']['ocp_bearing_inflate']
        self.save_feature_txt = yaml_setting['imp_data_preprocess']['save_feature_txt']
        self.save_scan_txt = yaml_setting['imp_data_preprocess']['save_scan_txt']
        self.save_feature_img = yaml_setting['imp_data_preprocess']['save_feature_img']
        self.overwrite_data = yaml_setting['imp_data_preprocess']['overwrite_data']
        self.preprocess_nfeature = yaml_setting['imp_data_preprocess']['nfeatures']
        self.preprocess_contrastTh = yaml_setting['imp_data_preprocess']['contrastThreshold']
        self.preprocess_edgeTh = yaml_setting['imp_data_preprocess']['edgeThreshold']
        self.tf_bodyInW = np.array(yaml_setting['tf_bodyInW']).reshape((4, 4))
        self.tf_camInB = np.array(yaml_setting['tf_camInB']).reshape((4, 4))
        self.tf_tagInCam = np.array(yaml_setting['tf_tagInCam']).reshape((4, 4))
        self.tf_tagFaceInBody = np.array(yaml_setting['tf_tagFaceInBody']).reshape((4,4))
        self.tf_aprilTagInBody = np.array(yaml_setting['tf_aprilTagInBody']).reshape((4,4))
        self.wall_color = yaml_setting['airsim']['wall_color']
        self.wall_seg_id = yaml_setting['airsim']['wall_seg_id']
        self.marker_sampling_ratio = yaml_setting['marker_pose_samples']['sampling_ratio']
        self.marker_voxel_size = yaml_setting['marker_pose_samples']['voxel_size']
        self.marker_min_distance = yaml_setting['marker_pose_samples']['min_distance']
        self.marker_outlier_distance = yaml_setting['marker_pose_samples']['outlier_distance']
        self.feat_min_sim_score = yaml_setting['similar_features']['min_sim_score']
        self.feat_min_distance = yaml_setting['similar_features']['min_distance']
        self.feat_parallel_jobs = yaml_setting['similar_features']['parallel_jobs']
        self.feat_batch_size = yaml_setting['similar_features']['batch_size']
        self.feat_voxel_size = yaml_setting['similar_features']['voxel_size']
        # intelligent marker placement
        self.marker_prior_std = yaml_setting['marker_placement']['marker_prior_std']
        self.marker_meas_std = yaml_setting['marker_placement']['marker_meas_std']
        self.cam_prior_std = yaml_setting['marker_placement']['cam_prior_std']
        self.lmk_prior_std = yaml_setting['marker_placement']['lmk_prior_std']
        self.pix_std = yaml_setting['marker_placement']['pix_std']
        self.imp_max_dist = yaml_setting['marker_placement']['max_dist']
        self.imp_max_angle = yaml_setting['marker_placement']['max_angle']
        self.imp_marker_num = yaml_setting['marker_placement']['marker_num']
        self.mean_max_weights = yaml_setting['marker_placement']['mean_max_weights']
        self.imp_pkl_res_path = yaml_setting['marker_placement']['pkl_res_path']
        self.placed_marker_path = yaml_setting['marker_placement']['placed_marker_path']
        self.imp_ignore_features = yaml_setting['marker_placement']['ignore_features']
        self.marker_percentages = yaml_setting['marker_placement']['most_vis_marker_percentages']
        # train-test datasets
        self.eva_marker_nums = yaml_setting['evaluation_datasets']['marker_nums']
        self.imp_max_ratios = yaml_setting['evaluation_datasets']['imp_max_ratios']
        self.eva_random_case_ids = yaml_setting['evaluation_datasets']['rd_ids']
        self.eva_unif_contour_case_ids = yaml_setting['evaluation_datasets']['uc_ids']
        self.test_img_num = yaml_setting['evaluation_datasets']['test_img_num']
        self.eva_ref_res_pkl = yaml_setting['evaluation_datasets']['ref_res_pkl']
        # self.test_cam_trans_noise_scale = yaml_setting['evaluation_datasets']['test_cam_trans_noise_scale']
        # self.test_cam_theta_noise_scale = yaml_setting['evaluation_datasets']['test_cam_theta_noise_scale']
        self.test_sample_mean_weight_scales = yaml_setting['evaluation_datasets']['test_sample_mean_weight_scales']
        self.test_data_distributions = yaml_setting['evaluation_datasets']['test_data_distributions']
        self.tag_size_ratios = yaml_setting['evaluation_datasets']['size_ratios']
        self.tag_tf_perturbations = yaml_setting['evaluation_datasets']['tf_perturb']
        # performance evaluation
        self.VLAD_visual_words = yaml_setting['performance_evaluation']['VLAD_visual_words']
        self.ball_tree_leaf_size = yaml_setting['performance_evaluation']['ball_tree_leaf_size']
        self.nearest_neighbor_imgs = yaml_setting['performance_evaluation']['nearest_neighbor_imgs']
        self.perf_parallel_jobs = yaml_setting['performance_evaluation']['parallel_jobs']
        self.perf_trans_angle_max = yaml_setting['performance_evaluation']['trans_angle_max']
        self.tag_family = yaml_setting['performance_evaluation']['tag_family']
        self.tag_size = yaml_setting['performance_evaluation']['tag_size']
        self.tag_loc_percent = yaml_setting['performance_evaluation']['tag_loc_percent']
        self.dft_sift = yaml_setting['performance_evaluation']['dft_sift_params']
        self.trans_tol = yaml_setting['performance_evaluation']['trans_tol']
        self.rad_tol = yaml_setting['performance_evaluation']['rad_tol']
        # generating marker placements for baseline methods
        if "baseline_method" in yaml_setting:
            if "unif_contour_min_distance" in yaml_setting["baseline_method"]:
                self.unif_contour_min_distance = yaml_setting["baseline_method"]["unif_contour_min_distance"]

def imp_setting(file_path=None) -> ParamStruct:
    if file_path is None:
        # arguments
        print('Number of arguments:', len(sys.argv), 'arguments.')
        print('Argument List:', str(sys.argv))

        # load yaml file
        assert len(sys.argv) >= 2
        file_path = sys.argv[1]
    else:
        if len(sys.argv) >= 2:
            file_path = sys.argv[1]
    with open(file_path, 'r') as stream:
        try:
            from yaml import CLoader as Loader, CDumper as Dumper
        except ImportError:
            from yaml import Loader, Dumper
        yaml_setting = yaml.load(stream, Loader=Loader)
    yaml_setting = ParamStruct(yaml_setting)
    return yaml_setting

def parseImageName(names: List[str]):
    rgb_name, dep_name, seg_name, sur_name = [None]*4
    for n in names:
        if n[:3] == "RGB":
            rgb_name = n
        elif n[:3] == "DEP":
            dep_name = n
        elif n[:3] == "SEG":
            seg_name = n
        elif n[:3] == "SUR":
            sur_name = n
        else:
            raise ValueError("Known image name types.")
    return rgb_name, dep_name, seg_name, sur_name

def write_params(write_dir, params: ParamStruct):
    out_yaml = os.path.join(write_dir, "imp_setting.yaml")
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
    if not os.path.exists(out_yaml):
        with open(out_yaml, "w+") as stream:
            try:
                from yaml import CLoader as Loader, CDumper as Dumper
            except ImportError:
                from yaml import Loader, Dumper
            yaml.dump(params.setting_dict, stream=stream, Dumper=Dumper)


def compute_loc_rate(err_file_path, t_th, angle_th):
    data = np.loadtxt(err_file_path)
    mask = (data[:, 0] < t_th) & (data[:, 1] < angle_th)
    return np.sum(mask)/len(data)


