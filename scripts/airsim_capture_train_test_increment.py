# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
import json
import pickle
import random
from typing import Union, List

import airsim
import pprint
import os
import time
import numpy as np
import yaml
from matplotlib import pyplot as plt, markers
from tqdm import tqdm

from optimized_marker_placement import ResultCase, KpScanData, extract_marker_poses, save_fig
from utils import imp_setting, ParamStruct, Pose3, Pose3OnXY


def generate_airsim_data(client: airsim.VehicleClient, work_dir,
                         cams_world_xyt: Union[list, np.ndarray], xlim, ylim, save_depth):
    # save data distribution
    fig, ax = plt.subplots(figsize=(8, int(8*(ylim[1]-ylim[0]) / (xlim[1]-xlim[0]))))
    bins = 30
    h = ax.hist2d(cams_world_xyt[:, 0], cams_world_xyt[:, 1], bins, range=[xlim, ylim], cmin=.9)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect('equal', 'box')
    ax.set_title("Data point distribution")
    # plt.show()
    fig.colorbar(h[3], ax=ax)
    fig.savefig(os.path.join(work_dir, "hist2d.png"),dpi=400)
    plt.close(fig)

    id_xyt = np.hstack([np.arange(len(cams_world_xyt)).reshape((-1, 1)), cams_world_xyt])
    np.savetxt(os.path.join(work_dir, "cams_xyt.txt"), X=id_xyt)

    v_name = "myCam"
    img_dir = os.path.join(work_dir, "images")
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    # print("Created data folder %s" % work_dir)
    gt_str = "\t".join(["VehicleName", "TimeStamp", "POS_X", "POS_Y", "POS_Z", "Q_W", "Q_X", "Q_Y", "Q_Z", "ImageFile", "ImgFolder"])

    sim_rec_file = os.path.join(work_dir, "airsim_rec.txt")
    exist_rec_data = None
    if os.path.exists(sim_rec_file):
        exist_rec_data = np.loadtxt(sim_rec_file, dtype=str, delimiter='\t')
    else:
        with open(sim_rec_file, "w+") as f_stream:
            f_stream.writelines(gt_str)
        gt_str = ""

    img_cnt = 0
    for cam_id in tqdm(range(len(cams_world_xyt))):
        # print(f"recording {cam_id}/{len(xyt_list)} images for case {folder_name}")

        rgb_name = "RGB" + str(img_cnt) + ".png"
        dep_name = "DEP" + str(img_cnt) + ".pfm"
        rgb_path = os.path.join(img_dir, rgb_name)
        dep_path = os.path.join(img_dir, dep_name)
        img_names = [rgb_name]
        if save_depth or os.path.exists(dep_path):
            img_names.append(dep_name)

        responses = []
        if not os.path.exists(rgb_path):
            responses.append(airsim.ImageRequest("0", airsim.ImageType.Scene))
        if save_depth and not os.path.exists(dep_path):
            responses.append(airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, pixels_as_float=True))

        # time_stamp = 0
        # px, py, pz, qw, qx, qy, qz = sim_pose.position.x_val, sim_pose.position.y_val, sim_pose.position.z_val,\
        #                              sim_pose.orientation.w_val,sim_pose.orientation.x_val,sim_pose.orientation.y_val,sim_pose.orientation.z_val
        # print("v pose:",px, py, pz, qw, qx, qy, qz)

        # sim_v_pose = Pose3.by_vec(px, py, pz, qw, qx, qy, qz)
        # sim_cam_pose = sim_v_pose * camInBody
        # px, py, pz, qw, qx, qy, qz = sim_cam_pose.transQuat
        if len(responses) > 0:
            cam_x, cam_y, cam_t = cams_world_xyt[cam_id, 0], -cams_world_xyt[cam_id, 1], -cams_world_xyt[cam_id, 2]
            # note that the theta x is in the airsim NED coordinate; negative x is positive is our NWU frame
            sim_pose = airsim.Pose(airsim.Vector3r(cam_x, cam_y, 0), airsim.to_quaternion(0, 0, cam_t))
            client.simSetVehiclePose(sim_pose,
                                     True)
            time.sleep(0.05)

            responses = client.simGetImages(responses)
            if None in responses or responses is None:
                print("None appears so re-do.")
                continue
            time_stamp = responses[-1].time_stamp
            px, py, pz, qw, qx, qy, qz = responses[-1].camera_position.x_val, responses[-1].camera_position.y_val, \
                                         responses[-1].camera_position.z_val,responses[-1].camera_orientation.w_val,\
                                         responses[-1].camera_orientation.x_val,responses[-1].camera_orientation.y_val,\
                                         responses[-1].camera_orientation.z_val
            # print("c pose:", px, py, pz, qw, qx, qy, qz)
            for i, response in enumerate(responses):
                if response.pixels_as_float:
                    airsim.write_pfm(dep_path, airsim.get_pfm_array(response))
                else:
                    airsim.write_file(rgb_path, response.image_data_uint8)
        # else:
        #     responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene)])
        #     if None in responses or responses is None:
        #         print("None appears so re-do.")
        #         continue
        #     time_stamp = responses[-1].time_stamp
        #     px, py, pz, qw, qx, qy, qz = responses[-1].camera_position.x_val, responses[-1].camera_position.y_val, \
        #                                  responses[-1].camera_position.z_val,responses[-1].camera_orientation.w_val,\
        #                                  responses[-1].camera_orientation.x_val,responses[-1].camera_orientation.y_val,\
        #                                  responses[-1].camera_orientation.z_val

            if exist_rec_data is None:
                gt_str += "\n"
                img_str = ';'.join(img_names)
                cur_line = "\t".join([v_name, str(time_stamp), str(px),
                                      str(py), str(pz),
                                      str(qw), str(qx),
                                      str(qy), str(qz), img_str, img_dir])
                gt_str += cur_line
                with open(os.path.join(work_dir, "airsim_rec.txt"), "a") as f:
                    f.writelines(gt_str)
                gt_str = ""
            else:
                print(f"updating cam pose {cam_id}: "+" ".join(img_names))
                # exist_rec_data[cam_id + 1, 9:11] = [";".join(img_names), img_dir]
        # time.sleep(.1)
        img_cnt += 1

def update_airsim_data(client: airsim.VehicleClient, cams_world_xyt, update_cam_idx, base_data_arr, work_dir, xlim, ylim, save_depth):
    if len(update_cam_idx) > 0:
        fig, ax = plt.subplots(figsize=(8, int(8 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))))
        bins = 30
        h = ax.hist2d(cams_world_xyt[update_cam_idx, 0], cams_world_xyt[update_cam_idx, 1], bins, cmin=.9, range=[xlim, ylim])
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal', 'box')
        ax.set_title("Data point distribution")
        # plt.show()
        fig.colorbar(h[3], ax=ax)
        fig.savefig(os.path.join(work_dir, "hist2d.png"), dpi=400)
        plt.close(fig)
        id_xyt = np.hstack([np.array(update_cam_idx).reshape((-1, 1)), cams_world_xyt[update_cam_idx]])
        np.savetxt(os.path.join(work_dir, "cams_xyt.txt"), X=id_xyt)

        img_dir = os.path.join(work_dir, "images")
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

        for cam_id in tqdm(update_cam_idx):
            airsim_x, airsim_y, airsim_t = cams_world_xyt[cam_id, 0], -cams_world_xyt[cam_id, 1], -cams_world_xyt[cam_id, 2]
            # note that the theta x is in the airsim NED coordinate; negative x is positive is our NWU frame
            # print(f"recording {cam_id}/{len(xyt_list)} images for case {folder_name}")

            rgb_name = "RGB" + str(cam_id) + ".png"
            dep_name = "DEP" + str(cam_id) + ".pfm"
            rgb_path = os.path.join(img_dir, rgb_name)
            dep_path = os.path.join(img_dir, dep_name)
            img_names = [rgb_name]
            if save_depth or os.path.exists(dep_path):
                img_names.append(dep_name)

            responses = []
            if not os.path.exists(rgb_path):
                responses.append(airsim.ImageRequest("0", airsim.ImageType.Scene))
            if save_depth and not os.path.exists(dep_path):
                responses.append(airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, pixels_as_float=True))

            # px, py, pz, qw, qx, qy, qz = sim_pose.position.x_val, sim_pose.position.y_val, sim_pose.position.z_val, \
            #                              sim_pose.orientation.w_val, sim_pose.orientation.x_val, sim_pose.orientation.y_val, sim_pose.orientation.z_val
            # sim_v_pose = Pose3.by_vec(px, py, pz, qw, qx, qy, qz)
            # sim_cam_pose = sim_v_pose * camInBody
            # px, py, pz, qw, qx, qy, qz = sim_cam_pose.transQuat
            if len(responses) > 0:
                sim_pose = airsim.Pose(airsim.Vector3r(airsim_x, airsim_y, 0), airsim.to_quaternion(0, 0, airsim_t))
                client.simSetVehiclePose(sim_pose,
                                         True)
                time.sleep(0.05)
                responses = client.simGetImages(responses)
                if None in responses or responses is None:
                    print("None appears so re-do.")
                    continue
                # px, py, pz, qw, qx, qy, qz = responses[-1].camera_position.x_val, responses[-1].camera_position.y_val, \
                #                              responses[-1].camera_position.z_val, responses[
                #                                  -1].camera_orientation.w_val, \
                #                              responses[-1].camera_orientation.x_val, responses[
                #                                  -1].camera_orientation.y_val, \
                #                              responses[-1].camera_orientation.z_val
                for i, response in enumerate(responses):
                    if response.pixels_as_float:
                        airsim.write_pfm(dep_path, airsim.get_pfm_array(response))
                    else:
                        airsim.write_file(rgb_path, response.image_data_uint8)
            # else:
            #     responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene)])
            #     if None in responses or responses is None:
            #         print("None appears so re-do.")
            #         continue
            #     px, py, pz, qw, qx, qy, qz = responses[-1].camera_position.x_val, responses[-1].camera_position.y_val, \
            #                                  responses[-1].camera_position.z_val, responses[
            #                                      -1].camera_orientation.w_val, \
            #                                  responses[-1].camera_orientation.x_val, responses[
            #                                      -1].camera_orientation.y_val, \
            #                                  responses[-1].camera_orientation.z_val
            # note the 0th line is the header
            base_data_arr[cam_id + 1, 9:11] = [";".join(img_names), img_dir]
            # time.sleep(.1)
    np.savetxt(fname=os.path.join(work_dir, "airsim_rec.txt"),X=base_data_arr, delimiter="\t", fmt="%s")

def train_test_data_airsim_xyt(params: ParamStruct, obs_pts: np.ndarray):
    angle_num = params.fine_angular_resolution
    # negative airsim theta in the NED frame is positive in our NWU frame
    angle_list = -np.linspace(0, 2 * np.pi, params.fine_angular_resolution, endpoint=False)
    train_data_xyt = np.zeros((len(obs_pts) * params.fine_angular_resolution, 3))
    for i, dot in enumerate(obs_pts):
        train_data_xyt[i * angle_num:(i + 1) * angle_num, 0] = dot[0]
        train_data_xyt[i * angle_num:(i + 1) * angle_num, 1] = dot[1]
        train_data_xyt[i * angle_num:(i + 1) * angle_num, 2] = angle_list.copy()

    test_data_xyt = {}
    test_data_dist = params.test_data_distributions
    test_cam_trans_noise_scale = params.grid_size
    test_cam_theta_noise_scale = 2*np.pi/params.fine_angular_resolution
    for suffix in test_data_dist:
        if suffix == 'unif':
            # uniformly sample free dots
            np.random.seed(params.random_seed)  # ensure consistent rd samples across runs
            random.seed(params.random_seed)
            xy_idx = np.random.choice(np.arange(len(obs_pts)),
                                      size=params.test_img_num)
            xy_list = obs_pts[xy_idx] + \
                      (np.random.random((params.test_img_num, 2)) - .5) * test_cam_trans_noise_scale
            theta_list = np.random.choice(angle_list,
                                          size=params.test_img_num) + \
                         (np.random.random(params.test_img_num) - .5) * test_cam_theta_noise_scale
            xyt_list = np.hstack((xy_list, theta_list.reshape((-1, 1))))
            test_data_xyt[suffix] = xyt_list
        elif suffix == 'weighted':
            test_data_xyt[suffix] = {}
            # weight test image samples by entropy
            pkl_res_file = params.eva_ref_res_pkl
            with open(pkl_res_file, "rb") as stream:
                imp_result_case = pickle.load(stream)
                assert isinstance(imp_result_case, ResultCase)
            cam_dots, entropy_arr = imp_result_case.Points, imp_result_case.EntropyMap
            pose0_inv = Pose3(params.tf_bodyInW).inverse()  # note we need to convert cam_dots back to NED frame
            cam_dots = pose0_inv.mat @ np.hstack([cam_dots, np.ones((len(cam_dots), 1))]).T
            cam_dots = cam_dots[:2, :].T
            cam_xyt = np.zeros((len(cam_dots) * angle_num, 3))
            for i, dot in enumerate(cam_dots):
                cam_xyt[i * angle_num:(i + 1) * angle_num, 0] = dot[0]
                cam_xyt[i * angle_num:(i + 1) * angle_num, 1] = dot[1]
                cam_xyt[i * angle_num:(i + 1) * angle_num, 2] = angle_list.copy()
            weights = (entropy_arr - np.min(entropy_arr)).flatten()
            for mean_weight_scale in params.test_sample_mean_weight_scales:
                np.random.seed(params.random_seed)  # ensure consistent rd samples across runs
                random.seed(params.random_seed)
                regularized_weights = weights + np.mean(weights) * mean_weight_scale + .001  # avoid vanishing weights
                regularized_weights = regularized_weights / sum(regularized_weights)
                cam_idx = random.choices(np.arange(len(cam_xyt)), weights=regularized_weights, k=params.test_img_num)
                xyt_list = cam_xyt[cam_idx]
                xyt_list[:, :2] += (np.random.random((params.test_img_num, 2)) - .5) * test_cam_trans_noise_scale
                xyt_list[:, 2] += (np.random.random(params.test_img_num) - .5) * test_cam_theta_noise_scale
                test_data_xyt[suffix][mean_weight_scale] = xyt_list
    return train_data_xyt, test_data_xyt

def poseIdInMarkerFOV(m_xyt, cams_xyt, scan_points, params):
    m_x_vec = np.array([np.cos(m_xyt[-1]), np.sin(m_xyt[-1])])
    m_y_vec = np.array([-np.sin(m_xyt[-1]), np.cos(m_xyt[-1])])

    cams_x_vec = np.array([np.cos(cams_xyt[:, -1]), np.sin(cams_xyt[:, -1])])
    cams_y_vec = np.array([-np.sin(cams_xyt[:, -1]), np.cos(cams_xyt[:, -1])])

    cam_m_diff = cams_xyt[:, :2] - m_xyt[:2]
    cams_x_in_m = np.dot(cam_m_diff, m_x_vec)
    cams_y_in_m = np.dot(cam_m_diff, m_y_vec)

    m_x_in_cams = np.diag(-cam_m_diff @ cams_x_vec)
    m_y_in_cams = np.diag(-cam_m_diff @ cams_y_vec)

    bearings_in_m = np.arctan2(cams_y_in_m, cams_x_in_m)
    bearings_in_cams = np.arctan2(m_y_in_cams, m_x_in_cams)
    ranges = np.linalg.norm(cam_m_diff, axis=1)

    safe_factor = 4/3

    # first filtering: cam and marker are co-visible
    infov = (ranges < params.imp_max_dist*safe_factor) & (abs(bearings_in_cams) < params.imp_max_angle*safe_factor * np.pi / 180) & (abs(bearings_in_m) < 4 * np.pi*safe_factor / 9)
    idx = np.arange(len(cams_xyt))[infov]

    scan_points_in_m = scan_points - m_xyt[:2]
    scan_ranges = np.linalg.norm(scan_points_in_m, axis=1)
    close_scan_idx = np.where(scan_ranges < .05)[0]

    scan_x_in_m = np.dot(scan_points_in_m, m_x_vec)
    scan_y_in_m = np.dot(scan_points_in_m, m_y_vec)
    scan_bearings = np.arctan2(scan_y_in_m, scan_x_in_m)

    idxInFov = []
    # analyze occlusion
    for i in idx:
        b_diff = abs(bearings_in_m[i] - scan_bearings)
        lo_b_diff = np.where(b_diff < .05)[0]
        lo_b_diff = [j for j in lo_b_diff if j not in close_scan_idx]
        if len(lo_b_diff) == 0 or min(scan_ranges[lo_b_diff])*1.05 > ranges[i]:
            idxInFov.append(i)
    return idxInFov

def poseIdInMarkersFOV(marker_world_xyt, cam_world_xyt, scan_points, params: ParamStruct, save_fov_plot=True, prefix="fovCheck",
                       xlim=None, ylim=None):
    cam_idx_list = []
    tmp_path = os.path.join(params.data_dir, "temp")
    if save_fov_plot:
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
    for m_xyt in marker_world_xyt:
        cams_idx = poseIdInMarkerFOV(m_xyt, cam_world_xyt, scan_points, params)
        cam_idx_list.append(cams_idx)
        # plotting
        if save_fov_plot:
            fig, ax = plt.subplots()
            if len(cams_idx) > 0:
                ax.hist2d(cam_world_xyt[cams_idx, 0], cam_world_xyt[cams_idx, 1], bins= 30, range=[xlim,ylim], cmin=.9)
                ax.scatter(cam_world_xyt[cams_idx, 0], cam_world_xyt[cams_idx, 1], c='blue', s=5)
            ax.scatter(scan_points[:, 0], scan_points[:, 1], c='red', s=1)
            x, y, theta = m_xyt
            arrow = markers.MarkerStyle(marker='$\u2193$')  # Downwards arrow in Unicode: ↓
            arrow._transform = arrow.get_transform().rotate_deg(90 + theta * 180 / np.pi)
            ax.scatter((x), (y), marker=arrow, c="black", s=100)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if save_fov_plot:
                save_fig(fig, tmp_path, prefix)
            plt.close(fig)
    return cam_idx_list

def copy_ref_airsim_file(ref_folder, path_col="ImgFolder", save_depth=False):
    ref_data = np.loadtxt(os.path.join(ref_folder, "airsim_rec.txt"), delimiter='\t', dtype=object)
    if ref_data[0,-1] != path_col:
        # the first image is RGB
        for i in range(0, len(ref_data)-1):
            if save_depth:
                ref_data[i+1, 9] = f"RGB{i}.png;DEP{i}.pfm"
            else:
                ref_data[i+1, 9] = f"RGB{i}.png"
        paths = [path_col]
        paths += [os.path.join(ref_folder,"images")] * (len(ref_data)-1)
        ref_data = np.append(ref_data, np.array(paths).reshape((-1, 1)), axis=1)
    return ref_data

def get_update_cam_idx(case2distPoseId, ref_marker_num, m_num, cur_data_type):
    cur_cams_ids = []
    if c_name in case2distPoseId:
        for tmp_i in range(ref_marker_num, m_num):
            cur_cams_ids += case2distPoseId[c_name][cur_data_type][tmp_i]
        cur_cams_ids = list(set(cur_cams_ids))
    return cur_cams_ids


if __name__ == "__main__":
    params = imp_setting("omp_settings.yaml")
    incremental_data_generation = True

    # pre-analysis for incremental recording
    no_marker_case = "NoMarker"
    dft_pct = 90
    if 0 in params.eva_marker_nums:
        case_names = [no_marker_case]
    else:
        case_names = []
    case_names += [f"Pct{i}Marker" for i in params.marker_percentages]
    case_names += [f"Uc{i}Marker" for i in params.eva_unif_contour_case_ids]
    case_names += [f"Rd{i}Marker" for i in params.eva_random_case_ids]

    free_pt_file = os.path.join(params.data_dir, "free_dot.txt")
    if os.path.exists(free_pt_file):
        obs_pts = np.loadtxt(free_pt_file)
    else:
        raise ValueError("No free_dot.txt file for placing cameras.")

    # fix random seeds
    xlim = [params.grid_x_offset, params.grid_x_offset + params.grid_x_len]
    ylim = [params.grid_y_offset, params.grid_y_offset + params.grid_y_len]

    # get the scan point map from the free dot data
    scan_map_path = os.path.join(params.data_dir, "scan_point_map.txt")
    if not os.path.exists(scan_map_path):
        pickle_file = os.path.join(params.data_dir, 'free_dots_data.pkl')
        assert os.path.exists(pickle_file)
        with open(pickle_file, "rb") as f:
            all_cam_dots = pickle.load(f)
        assert isinstance(all_cam_dots[0], KpScanData)
        extract_marker_poses(data=all_cam_dots, wall_color=None,
                             sampling_ratio=params.marker_sampling_ratio,
                             voxel_size=params.marker_voxel_size,
                             max_depth=params.depth_max,
                             use_data_seg=False,
                             use_data_nml=True,
                             min_distance=params.marker_min_distance,
                             marker_outlier_distance=params.marker_outlier_distance,
                             save_fig_path=os.path.join(params.data_dir, "scan_point_map.png"),
                             save_scan_path=scan_map_path)
    scan_points = np.loadtxt(scan_map_path)
    train_airsim_xyt, test_airsim_xyt = train_test_data_airsim_xyt(params, obs_pts)

    pp = pprint.PrettyPrinter(indent=4)
    v_name = "myCam"
    cam_id = params.cam_name
    # connection
    client = airsim.VehicleClient()
    client.confirmConnection()
    eva_folder = os.path.join(params.data_dir, "evaluation")
    if not os.path.exists(eva_folder):
        os.mkdir(eva_folder)

    # camera info
    camera_info = client.simGetCameraInfo(str(cam_id))
    camera_distort = client.simGetDistortionParams(str(cam_id))
    p1 = pp.pformat(camera_info)
    p2 = pp.pformat(camera_distort)
    print("CameraInfo %s: %s" % (cam_id, p1))
    print("CameraDistort %s: %s" % (cam_id, p2))
    info_path = os.path.join(eva_folder, "camera_info.txt")
    with open(info_path, "w+") as f:
        f.write(pp.pformat(camera_info))
        f.write('\n')
        f.write(pp.pformat(camera_distort))

    setting_path = os.path.join(eva_folder, "airsim_setting.txt")
    with open(setting_path, "w+") as f:
        f.write(client.getSettingsString())

    out_yaml = os.path.join(eva_folder, "imp_setting.yaml")
    if not os.path.exists(out_yaml):
        with open(out_yaml, "w+") as stream:
            try:
                from yaml import CLoader as Loader, CDumper as Dumper
            except ImportError:
                from yaml import Loader, Dumper
            yaml.dump(params.setting_dict, stream=stream, Dumper=Dumper)

    # segmentations
    set_seg = True
    # IDs correspond to colors in this list:  # https://microsoft.github.io/AirSim/seg_rgbs.txt
    bg_ID = 0 # color [0, 0, 0]
    obj_ID = params.wall_seg_id # color [153, 108, 6]
    object_name_list = params.wall_labels
    if set_seg:
        print("Reset all object id")
        found = client.simSetSegmentationObjectID("[\w]*", 0, True)
        print("all object: %r" % (found))
        time.sleep(1.0)
        for idx, obj_name in enumerate(object_name_list):
            obj_name_reg = r"[\w]*" + obj_name + r"[\w]*"
            found = client.simSetSegmentationObjectID(obj_name_reg, obj_ID, True)
            print("%s: %r" % (obj_name, found))

    time.sleep(5.0)

    dist2cams_xyt = {}
    dist2cams_xyt['train'] = train_airsim_xyt
    dist2cams_xyt['train'][:, 1] *= -1
    dist2cams_xyt['train'][:, 2] *= -1
    for suffix in params.test_data_distributions:
        if suffix == 'unif':
            key_v = f'test_{suffix}'
            dist2cams_xyt[key_v] = test_airsim_xyt[suffix].copy()
            dist2cams_xyt[key_v][:, 1] *= -1
            dist2cams_xyt[key_v][:, 2] *= -1
        elif suffix == 'weighted':
            for mean_weight_scale in params.test_sample_mean_weight_scales:
                key_v = f'test_{suffix}_meanScale{mean_weight_scale}'
                dist2cams_xyt[key_v] = test_airsim_xyt[suffix][mean_weight_scale]
                dist2cams_xyt[key_v][:, 1] *= -1
                dist2cams_xyt[key_v][:, 2] *= -1

    pickle_file = os.path.join(eva_folder, "case2poseIdInDist.pkl")
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            case2distPoseId = pickle.load(f)
        for c_name in case_names:
            if c_name != no_marker_case:
                for d_dist, cams_world_xyt in dist2cams_xyt.items():
                    if c_name in case2distPoseId and d_dist in case2distPoseId[c_name]:
                        print(f"--{d_dist}")
                        marker2poseid = case2distPoseId[c_name][d_dist]
                        print(" ".join([str(len(line)) for line in marker2poseid]))
    else:
        case2distPoseId = {}

    maxMarkerNum = max(params.eva_marker_nums)
    case2marker_xyt = {}
    for c_name in case_names:
        if c_name != no_marker_case:
            print(f"--{c_name}")
            saved_markers = [f"tag{c_name}{i}" for i in range(maxMarkerNum)]
            marker_world_xyt = []
            for m_str in saved_markers:
                found = client.simGetObjectPose(m_str)
                # print("obj pose: %r" % (found))
                m_pose_airsim = Pose3.by_vec(found.position.x_val, found.position.y_val, found.position.z_val,
                                             found.orientation.w_val, found.orientation.x_val,
                                             found.orientation.y_val,
                                             found.orientation.z_val)
                m_pose_world = Pose3(params.tf_bodyInW) * m_pose_airsim * Pose3(
                    params.tf_tagFaceInBody)  # * Pose3(params.tf_camInB) * Pose3(params.tf_tagInCam)
                x, y, theta = Pose3OnXY(m_pose_world)
                marker_world_xyt.append([x, y, theta])
            case2marker_xyt[c_name] = marker_world_xyt

    for c_name, marker_world_xyt in case2marker_xyt.items():
        print(f"FOV analysis for {c_name}")
        if c_name not in case2distPoseId:
            case2distPoseId[c_name] = {}
        for d_dist, cams_world_xyt in dist2cams_xyt.items():
            print(f"--{d_dist}")
            if d_dist not in case2distPoseId[c_name]:
                print("updating FOV analysis")
                case2distPoseId[c_name][d_dist] = poseIdInMarkersFOV(marker_world_xyt, cams_world_xyt,
                                                                     scan_points, params,
                                                                     save_fov_plot=True, prefix=f'{c_name}_{d_dist}',
                                                                     xlim=xlim, ylim=ylim)
            print(" ".join([str(len(line)) for line in case2distPoseId[c_name][d_dist]]))
    with open(pickle_file, "wb") as f:
        pickle.dump(case2distPoseId, f)

    # sorted_marker_nums
    sorted_marker_nums = np.sort(params.eva_marker_nums)
    for c_name in case_names:
        print(f"Process {c_name}")
        c_dir = os.path.join(eva_folder, c_name)
        if not os.path.exists(c_dir):
            os.mkdir(c_dir)
        for m_idx, m_num in enumerate(sorted_marker_nums):
            record_train = False
            record_test = False
            case_folder = None
            train_ref_folder = None
            test_ref_folder = None
            update_all_cams = False
            ref_marker_num = None
            if m_num == 0 and c_name == no_marker_case:
                record_test = True
                record_train = True
                case_folder = c_dir
                train_ref_folder = os.path.join(params.data_dir, "free_dots_scan_images")
                ref_marker_num = 0
                # if os.path.exists(os.path.join(case_folder, cur_data_type))
            elif m_num !=0 and c_name != no_marker_case:
                record_test = True
                record_train = True
                case_folder = os.path.join(c_dir, f"{m_num}markers")
                if not os.path.exists(case_folder):
                    os.mkdir(case_folder)
                if (m_idx == 1 and sorted_marker_nums[0] == 0) or m_idx == 0:
                    # refer to the no marker case folders
                    train_ref_folder = os.path.join(eva_folder, no_marker_case, "train")
                    test_ref_folder = os.path.join(eva_folder, no_marker_case)
                    ref_marker_num = 0
                else:
                    ref_marker_num = sorted_marker_nums[m_idx - 1]
                    train_ref_folder = os.path.join(c_dir, f"{ref_marker_num}markers","train")
                    test_ref_folder = os.path.join(c_dir, f"{ref_marker_num}markers")
            else:
                continue

            print(f"marker num: {m_num}")

            if m_num == 0:
                saved_markers = []
                saved_holders = []
            else:
                saved_markers = [f"tag{c_name}{i}" for i in range(m_num)]
                saved_holders = [f"holder{c_name}{i}" for i in range(m_num)]
            if len(saved_markers) > 0:
                print("-moving and saving markers")
                for i, m_name in enumerate(saved_markers):
                    m_pose = client.simGetObjectPose(m_name)
                    m_pose.position.z_val = 0.0
                    success = client.simSetObjectPose(m_name, m_pose)
                    # print(f"Move {m_name} to 0: {success}")
                    if not success:
                        raise ValueError("cant simSetObjectPose")
                    h_name = saved_holders[i]
                    h_pose = client.simGetObjectPose(h_name)
                    h_pose.position.z_val = 0.0
                    success = client.simSetObjectPose(h_name, h_pose)
                    # print(f"Move {h_name} to 0: {success}")
                    if not success:
                        print(f"marker name: {m_name} pose: {m_pose}")
                        raise ValueError("cant simSetObjectPose")
                time.sleep(5.0)

                gt_str = "\t".join(["MarkerName", "POS_X", "POS_Y", "POS_Z", "Q_W", "Q_X", "Q_Y", "Q_Z"])
                for m_str in saved_markers:
                    found = client.simGetObjectPose(m_str)
                    # print("obj pose: %r" % (found))
                    gt_str += "\n"
                    gt_str += "\t".join([m_str, str(found.position.x_val),
                                         str(found.position.y_val), str(found.position.z_val),
                                         str(found.orientation.w_val), str(found.orientation.x_val),
                                         str(found.orientation.y_val), str(found.orientation.z_val)])
                with open(os.path.join(case_folder, "airsim_marker.txt"), "w+") as f:
                    f.writelines(gt_str)
                gt_str = ""

            for cur_data_type, cams_xyt in dist2cams_xyt.items():
                print(f"-{cur_data_type}")
                work_dir = os.path.join(case_folder, cur_data_type)
                if not os.path.exists(work_dir):
                    os.mkdir(work_dir)
                cur_ref_folder = None
                save_depth = False
                update_all_cams = False
                if cur_data_type == "train":
                    save_depth = True
                    if train_ref_folder is None:
                        cur_ref_folder = None
                    else:
                        cur_ref_folder = train_ref_folder
                else:
                    if test_ref_folder is None:
                        cur_ref_folder = None
                    else:
                        cur_ref_folder = os.path.join(test_ref_folder, cur_data_type)
                if cur_ref_folder is not None:# and (incremental_data_generation or (c_name==no_marker_case and cur_data_type=="train")):
                    base_rec_data = copy_ref_airsim_file(cur_ref_folder, save_depth=save_depth)
                    # get cam idx that will be updated
                    update_cam_idx = get_update_cam_idx(case2distPoseId, ref_marker_num, m_num, cur_data_type)
                    if len(update_cam_idx) > 0 and not incremental_data_generation:
                        update_cam_idx = np.arange(len(dist2cams_xyt[cur_data_type]))
                    update_airsim_data(client, dist2cams_xyt[cur_data_type], update_cam_idx, base_rec_data, work_dir, xlim, ylim, save_depth)
                else:
                    generate_airsim_data(client, work_dir, dist2cams_xyt[cur_data_type], xlim, ylim, save_depth)
            if len(saved_markers)>0:
                for i, m_name in enumerate(saved_markers):
                    m_pose = client.simGetObjectPose(m_name)
                    m_pose.position.z_val = -99
                    success = client.simSetObjectPose(m_name, m_pose)
                    # print(f"Reset {m_name} to {m_pose.position.z_val}: {success}")
                    if not success:
                        raise ValueError("cant simSetObjectPose")
                    h_name = saved_holders[i]
                    h_pose = client.simGetObjectPose(h_name)
                    h_pose.position.z_val = -99
                    success = client.simSetObjectPose(h_name, h_pose)
                    # print(f"Reset {h_name} to {h_pose.position.z_val}: {success}")
                    if not success:
                        raise ValueError("cant simSetObjectPose")
                time.sleep(5.0)