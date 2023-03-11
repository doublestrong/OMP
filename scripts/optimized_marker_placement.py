import pickle

import scipy.spatial.distance as sci_dist
from typing import Union, List
from dataclasses import dataclass
from tqdm import tqdm
import open3d as o3d
import os
import gtsam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as markers
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import cv2 as cv
from PIL import Image
import random

from utils import Pose3OnXY, Pose3, imp_setting, ParamStruct, parseImageName, read_pfm, px2coor, write_params, \
    depth_interpolation


def NED_cam_ray_bearing(w, focal_len):
    """
    On the X-Y plane, if x is pointing right, y is pointing down.
    The central axis of beams is the x axis.
    :param w: width of image, int
    :param focal_len: focal length in pixels
    :return: bearing and width of a beam in ray casting; bearing on the right is positive
    """
    c_w = w / 2.0
    ray_y_arr = np.arange(0, w, 1) - c_w + .5
    ray_bearing = np.arctan2(ray_y_arr, focal_len)
    ray_l_edge_arr = np.arange(0, w, 1) - c_w
    ray_l_edge_bearing = np.arctan2(ray_l_edge_arr, focal_len)
    ray_r_edge_bearing = ray_l_edge_bearing.copy()
    ray_r_edge_bearing[-1] = -ray_l_edge_bearing[0]
    ray_r_edge_bearing[:-1] = ray_l_edge_bearing[1:]
    return ray_bearing, abs(ray_r_edge_bearing - ray_l_edge_bearing)

def Pose3RangeBearingOnXY(pose1: Pose3, pose2: Pose3):
    pose2in1 = pose1.inverse() * pose2
    x, y = pose2in1.translation[:2]
    return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)


def defPriorPose3Factor(key: int, pose: Pose3, noise_sigmas: np.ndarray):
    # noise model: rpyxyz
    sam_pose = gtsam.Pose3(pose.mat)
    pose_nm = gtsam.noiseModel.Diagonal.Sigmas(noise_sigmas)
    return gtsam.PriorFactorPose3(key, sam_pose, pose_nm)


def defBetweenPose3Factor(key1: int, key2: int, pose: Pose3, noise_sigmas: np.ndarray):
    noisy_tf = gtsam.Pose3(pose.mat)
    pose_nm = gtsam.noiseModel.Diagonal.Sigmas(noise_sigmas)
    return gtsam.BetweenFactorPose3(key1, key2, noisy_tf, pose_nm)


def defPriorPoint3Factor(key: int, point: np.ndarray, noise_sigmas: np.ndarray):
    sam_pose = gtsam.Point3(point)
    pose_nm = gtsam.noiseModel.Diagonal.Sigmas(noise_sigmas)
    return gtsam.PriorFactorPoint3(key, sam_pose, pose_nm)

def visualize(fig, ax,
              camPoints: np.ndarray,
              entropy_list=None,
              placedMarkers: List[Pose3] = [],
              trialMarkers: List[Pose3] = [],
              vminmax=[],
              cmap='binary',
              placed_color="red",
              trial_color="blue",
              scan_points=None,
              plot_score=True,
              offset=0.5,
              text_annotate=True,
              dot_size=20):

    if entropy_list is not None and plot_score:
        if len(vminmax) > 0:
            res = ax.scatter(camPoints[:, 0], camPoints[:, 1], marker='.', s=dot_size,
                             c=entropy_list, cmap=cmap, vmin=vminmax[0], vmax=vminmax[1])
        else:
            res = ax.scatter(camPoints[:, 0], camPoints[:, 1], marker='.', s=dot_size,
                             c=entropy_list, cmap=cmap)
        fig.colorbar(res, ax=ax)

    if scan_points is not None:
        ax.scatter(scan_points[:, 0], scan_points[:, 1], c="black", s=.05)

    text_yx = []
    text_str = []

    for i, pose in enumerate(placedMarkers):
        x, y, theta = Pose3OnXY(pose)
        arrow = markers.MarkerStyle(marker='$\u2193$')  # Downwards arrow in Unicode: ↓
        arrow._transform = arrow.get_transform().rotate_deg(90 + theta * 180 / np.pi)
        ax.scatter((x), (y), marker=arrow, s=80, c=placed_color)
        text_yx.append([y-offset,x+offset])
        text_str.append(str(i+1))
        # ax.arrow(x, y, .45*np.cos(theta), .45*np.sin(theta), color=placed_color, shape='full', lw=0.8, length_includes_head=True, head_width=.2)
        if text_annotate:
            ax.text((x + 0), (y - offset), str(i + 1), c=placed_color)

    for i, pose in enumerate(trialMarkers):
        x, y, theta = Pose3OnXY(pose)
        arrow = markers.MarkerStyle(marker='$\u2193$')  # Downwards arrow in Unicode: ↓
        arrow._transform = arrow.get_transform().rotate_deg(90 + theta * 180 / np.pi)
        ax.scatter((x), (y), marker=arrow, s=80, c=trial_color)
        # ax.arrow(x, y, .45*np.cos(theta), .45*np.sin(theta), color=trial_color, shape='full', lw=0.8, length_includes_head=True, head_width=.2)
    return fig, ax


@dataclass
class KpScanData:
    Poses: List[Pose3]
    Keypoints: List[Union[np.ndarray, None]]
    Scans: List[np.ndarray]
    ScanSegs: List[np.ndarray]
    ScanNmls: List[np.ndarray]
    Descriptors: List[Union[np.ndarray, None]]


def cov2entropy(cov):
    return .5 * (np.log(np.linalg.det(2 * np.pi * cov)) + 1) # the constant D=6 does not matter since it won't affect the rank of entropy


@dataclass
class ResultCase:
    Points: np.ndarray
    PlacedMarkers: List[Pose3]
    TrialMarkers: List[Pose3]
    EntropyMap: np.ndarray
    PlacedIDs: Union[List, np.ndarray]


def entropyScore(entropy_map, mean_max_weights):
    if len(entropy_map.shape) == 2:
        mean_entropy = np.mean(entropy_map, axis=1)
        max_entropy = np.max(entropy_map, axis=1)
    else:
        mean_entropy = np.mean(entropy_map)
        max_entropy = np.max(entropy_map)
    return mean_max_weights[0] * mean_entropy + mean_max_weights[1] * max_entropy


def save_locscore_plot(entropy_map, run_dir, cam_points, placedMarkers, trailMarkers, xlim, ylim, vminmax=[],
                       prefix="score", scan_points=None):
    mean_entropy_list = np.mean(entropy_map, axis=1)
    fig, ax = plt.subplots(figsize=(8, int(8 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))))
    fig, ax = visualize(fig, ax, cam_points, -mean_entropy_list, placedMarkers, trailMarkers, vminmax, cmap="gray")
    if scan_points is not None:
        ax.scatter(scan_points[:, 0], scan_points[:, 1], c="red", s=.05)
    ax.set_title(
        f"Mean Localizability Score: {round(-np.mean(mean_entropy_list), 4)}")
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_aspect('equal', 'box')
    save_fig(fig, run_dir, prefix)
    plt.close(fig)


def save_markers_plot(run_dir, cam_points, marker_poses, xlim, ylim,
                       prefix="markers", scan_points=None):

    fig, ax = plt.subplots(figsize=(8, int(8 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))))
    for m_pose in marker_poses:
        x, y, theta = Pose3OnXY(m_pose)
        dx, dy = np.cos(theta), np.sin(theta)
        # ax.annotate("", xy=(x+dx, y+dy), xytext=(x, y),
        #             arrowprops=dict(arrowstyle="->"))
        ax.arrow(x, y, dx*.45, dy*.45, color="black", shape='full', lw=0.8, length_includes_head=True, head_width=.2)
    ax.scatter(cam_points[:, 0], cam_points[:, 1], c="blue", s=.3)
    if scan_points is not None:
        ax.scatter(scan_points[:, 0], scan_points[:, 1], c="red", s=.05)
    ax.set_aspect('equal')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    save_fig(fig, run_dir, prefix)
    plt.close(fig)

def save_info_gain_plot(gain_list, gain_score, run_dir, cam_points, placedMarkers, trailMarkers, xlim, ylim, vminmax=[],
                        prefix="gain",percentileAndGain=None, scan_points=None):
    fig, ax = plt.subplots(figsize=(8, int(8 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))))
    fig, ax = visualize(fig, ax, cam_points, gain_list, placedMarkers, trailMarkers, vminmax)
    if percentileAndGain is None:
        ax.set_title(
            f"Mean Information Gain: {round(gain_score, 4)}")
    else:
        ax.set_title(
            f"Mean: {round(gain_score, 2)}, {round(percentileAndGain[0], 1)} pclt: {round(percentileAndGain[1], 2)}")
    if scan_points is not None:
        ax.scatter(scan_points[:, 0], scan_points[:, 1], c="red", s=.05)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_aspect('equal', 'box')
    save_fig(fig, run_dir, prefix)
    plt.close(fig)


def informationScore(new_entropy_map, old_entropy_map, mean_max_weights, percentile=None):
    info_gain = old_entropy_map - new_entropy_map
    gain_mean_list = np.mean(info_gain, axis=1)
    gain_max_list = np.max(info_gain, axis=1)
    score_list = mean_max_weights[0] * gain_mean_list + mean_max_weights[1] * gain_max_list
    info_score = mean_max_weights[0] * np.mean(info_gain) + mean_max_weights[1] * np.max(info_gain)
    if percentile is None:
        return score_list, info_score
    else:
        return score_list, info_score, np.percentile(info_gain.flatten(), percentile)


class MarkerPlacement:
    def __init__(self, raw_data: List[KpScanData], marker_poses: List[Pose3],
                 kp_std, mark_std, pix_std, T_std, cam_std,
                 scale_noise=True, max_dist=10, max_angle=np.pi / 4,
                 focal_len=600, image_w=1200, image_h=900,
                 max_marker_angle=4 * np.pi / 9,
                 sim_feat_count=None, placed_id=None, ignore_features=False,scan_points=None):
        """
        :param raw_data:
        :param marker_poses:
        :param kp_std:
        :param mark_std:
        :param pix_std:
        :param T_std:
        :param cam_std:
        :param scale_noise:
        :param max_dist:
        :param max_angle:
        :param focal_len:
        :param image_w:
        :param image_h:
        :param max_marker_angle:
        :param sim_feat_count: list
        """
        self.data = raw_data
        self.cam_points = np.array([d.Poses[0].translation for d in self.data])
        self.cam_pose_num = len(self.cam_points) * len(self.data[0].Poses)
        self.marker_poses = marker_poses
        if placed_id is None:
            self.placed_idx = []
        else:
            self.placed_idx = placed_id[:]

        self.kp_std = kp_std
        self.mark_std = mark_std
        self.pix_std = pix_std
        self.cam_std = cam_std
        self.T_std = T_std
        self.scale_noise = scale_noise

        self.max_dist = max_dist
        self.max_angle = max_angle
        self.max_marker_angle = max_marker_angle
        self.image_width = len(self.data[0].Scans[0])
        assert self.image_width == image_w
        self.image_height = image_h
        self.focal_len = focal_len
        self.ray_bearing, self.ray_width = NED_cam_ray_bearing(image_w, focal_len)
        self.cam_model = gtsam.Cal3_S2(focal_len, focal_len, .0, int(image_w / 2), int(image_h / 2))
        mat = np.eye(4)
        mat[:3, 0] = [0, 1, 0]
        mat[:3, 1] = [0, 0, 1]
        mat[:3, 2] = [1, 0, 0]
        self.camInBody = Pose3(mat)
        self.markerIDInFOV, self.markerID2camIJ = self.computeFOV()
        self.markerID2camPercentage = {k: len(v) / self.cam_pose_num * 100.0 for k, v in self.markerID2camIJ.items()}
        self.sim_feat_count = sim_feat_count
        self.cachedEntropyMap = None
        self.ignore_features = ignore_features
        self.scan_points = scan_points
        print("Finish initialization.")

    def markerInFOV(self, cam: Pose3, lmk: Pose3, cam_pt_idx: int, cam_orient_idx: int):
        # currently, we assume cam and lmk are on the plane parallel to the ground
        # cam2 = Pose3toPose2(cam)
        # lmk2 = Pose3toPose2(lmk)
        # dist, bearing = cam2.range_and_bearing(lmk2.translation)
        # dist2, bearing2 = lmk2.range_and_bearing(cam2.translation)

        # CAM is an NED frame
        dist, bearing = Pose3RangeBearingOnXY(cam, lmk)
        dist2, bearing2 = Pose3RangeBearingOnXY(lmk, cam)
        if dist < self.max_dist and abs(bearing) < self.max_angle and abs(bearing2) < self.max_marker_angle:
            scan = self.data[cam_pt_idx].Scans[cam_orient_idx]
            scan = cam.inverse().tf_points(scan)
            b_diff = abs(bearing - self.ray_bearing)
            b_idx = np.argmin(b_diff)
            b_range = np.linalg.norm(scan[b_idx, :2])
            return dist < b_range + .05 and b_diff[b_idx] < .1
        return False

    def computeFOV(self):
        markerIDInFOV = []
        markerID2camIJ = {}
        for i in range(len(self.data)):
            mIDatCamPt = []
            for j, cam_pose in enumerate(self.data[i].Poses):
                mIDs = []
                for k, m_pose in enumerate(self.marker_poses):
                    if k not in markerID2camIJ:
                        markerID2camIJ[k] = set()
                    if self.markerInFOV(cam_pose, m_pose, i, j):
                        mIDs.append(k)
                        markerID2camIJ[k].add((i, j))
                mIDatCamPt.append(mIDs)
            markerIDInFOV.append(mIDatCamPt)
        return markerIDInFOV, markerID2camIJ

    def placeMarker(self, newMarkerID: int = None, entropyMap: np.ndarray = None):
        if newMarkerID is not None:
            assert newMarkerID not in self.placed_idx
            self.placed_idx.append(newMarkerID)
        if entropyMap is not None:
            self.cachedEntropyMap = entropyMap.copy()

    def singleCamPoseAnalysis(self, cam_pt_idx: int, cam_orient_idx: int, newMarkerID=None):
        markerIDs = self.markerIDInFOV[cam_pt_idx][cam_orient_idx]
        vis_markerIDs = []
        if newMarkerID is not None:
            if newMarkerID in markerIDs:
                vis_markerIDs.append(newMarkerID)
            elif self.cachedEntropyMap is not None:  # use cached entropy map
                return self.cachedEntropyMap[cam_pt_idx, cam_orient_idx]
        vis_markerIDs += list(set(self.placed_idx).intersection(set(markerIDs)))
        cam_var = gtsam.Symbol('X', 0).key()
        cam_pose = self.data[cam_pt_idx].Poses[cam_orient_idx] * self.camInBody

        cam_model = gtsam.Cal3_S2(self.focal_len, self.focal_len, .0, int(self.image_width / 2),
                                  int(self.image_height / 2))
        camera = gtsam.PinholeCameraCal3_S2(gtsam.Pose3(cam_pose.mat), cam_model)
        cam_prior = defPriorPose3Factor(cam_var, cam_pose, self.cam_std)

        # create bearing factors
        kps = self.data[cam_pt_idx].Keypoints[cam_orient_idx]
        if kps is None:
            lmks = []
        else:
            lmks = kps.copy()

        # camera noise
        cam_noise = gtsam.noiseModel.Isotropic.Sigma(2, self.pix_std)

        fg = gtsam.NonlinearFactorGraph()
        fg.add(cam_prior)
        values = gtsam.Values()
        values.insert(cam_var, gtsam.Pose3(cam_pose.mat))

        # print(f"adding {len(lmks)} lmks")
        if not self.ignore_features:
            for i, lmk in enumerate(lmks):
                lmk_var = gtsam.Symbol('L', i).key()
                if self.sim_feat_count is not None:
                    sim_feat_num = self.sim_feat_count[cam_pt_idx][cam_orient_idx][i]
                    lmk_prior = defPriorPoint3Factor(lmk_var, lmk, self.kp_std * np.sqrt(1 + sim_feat_num))
                else:
                    lmk_prior = defPriorPoint3Factor(lmk_var, lmk, self.kp_std)
                measurement = camera.project(lmk)
                b_factor = gtsam.GenericProjectionFactorCal3_S2(
                    measurement, cam_noise, cam_var, lmk_var, cam_model)
                fg.add(lmk_prior)
                values.insert(lmk_var, gtsam.Point3(lmks[i, :]))
                fg.push_back(b_factor)

        # groundtruth bearing
        for i in vis_markerIDs:
            mark = self.marker_poses[i]
            mark_var = gtsam.Symbol('M', i).key()
            rel_pose = mark.inverse() * cam_pose
            mark_prior = defPriorPose3Factor(mark_var, rel_pose, self.mark_std)
            T_factor = defBetweenPose3Factor(mark_var, cam_var, rel_pose, self.T_std)
            fg.add(mark_prior)
            fg.add(T_factor)
            values.insert(mark_var, gtsam.Pose3(mark.mat))

        marginals = gtsam.Marginals(fg, values)
        ga_cov = marginals.marginalCovariance(cam_var)
        return cov2entropy(ga_cov)

    def singleCamDotAnalysis(self, cam_pt_idx: int, newMarkID=None):
        num_cam_angles = len(self.data[0].Poses)
        entropy_arr = np.zeros(num_cam_angles)
        for j in range(num_cam_angles):
            entropy_arr[j] = self.singleCamPoseAnalysis(cam_pt_idx, j, newMarkID)
        return entropy_arr

    def allCamAnalysis(self, newMarkID=None, parallel_jobs=1):
        if newMarkID is None:
            print("Analyzing the current environment.")
        else:
            print(f"Analyzing for placing marker {newMarkID}.")
        num_cam_angles = len(self.data[0].Poses)
        num_cam_dots = len(self.data)
        if parallel_jobs < 2:
            if newMarkID is None or self.cachedEntropyMap is None:
                entropy_arr = np.zeros((num_cam_dots, num_cam_angles))
                for i in range(num_cam_dots):
                    for j in range(num_cam_angles):
                        entropy_arr[i, j] = self.singleCamPoseAnalysis(i, j, newMarkID)
            else:
                entropy_arr = self.cachedEntropyMap.copy()
                for i, j in self.markerID2camIJ[newMarkID]:
                    entropy_arr[i, j] = self.singleCamPoseAnalysis(i, j, newMarkID)
        else:
            entropy_arr = Parallel(n_jobs=parallel_jobs)(
                delayed(self.singleCamDotAnalysis)(i, newMarkID) for i in range(num_cam_dots))
            entropy_arr = np.array(entropy_arr)
        return entropy_arr

    def pickle_res(self, save_dir, entropy_map, trailIDs):
        cnt = 0
        file_path = os.path.join(save_dir, f"res{cnt}.pkl")
        while True:
            if os.path.exists(file_path):
                cnt += 1
                file_path = os.path.join(save_dir, f"res{cnt}.pkl")
            else:
                break
        res = ResultCase(self.cam_points,
                         [self.marker_poses[i] for i in self.placed_idx],
                         [self.marker_poses[i] for i in trailIDs],
                         entropy_map,
                         self.placed_idx)
        with open(file_path, "wb") as f:
            pickle.dump(res, f)

    def create_run_dir(self, save_dir):
        cnt = 0
        run_dir = os.path.join(save_dir, f"run{cnt}")
        while True:
            if os.path.exists(run_dir):
                cnt += 1
                run_dir = os.path.join(save_dir, f"run{cnt}")
            else:
                os.mkdir(run_dir)
                break
        return run_dir

    @property
    def placedMarkerPoses(self):
        return [self.marker_poses[id] for id in self.placed_idx]

    def most_visible_markers(self, percentage: Union[int, float]):
        unplaced_ids = [i for i in range(len(self.marker_poses)) if i not in self.placed_idx]
        vis_cam_percentage = [self.markerID2camPercentage[k] for k in unplaced_ids]
        cam_percentile = np.percentile(vis_cam_percentage, 100-percentage)
        selected_marker_ids = np.array(unplaced_ids)[vis_cam_percentage > cam_percentile]
        return selected_marker_ids, 100-cam_percentile

    def start_placing(self, mark_num, save_dir, params: ParamStruct, vminmax=[], marker_percentage=None):
        xlim = [params.grid_x_offset, params.grid_x_offset + params.grid_x_len]
        ylim = [params.grid_y_offset, params.grid_y_offset + params.grid_y_len]

        if marker_percentage is not None:
            # save params
            run_dir = self.create_run_dir(save_dir)
            write_params(run_dir, params)
            entropy_map = self.allCamAnalysis()
            self.placeMarker(None, entropy_map)
            self.pickle_res(run_dir, entropy_map, [])
            # plot mean entropy
            save_locscore_plot(entropy_map, run_dir, cam_points=self.cam_points,
                               placedMarkers=self.placedMarkerPoses, trailMarkers=[],
                               xlim=xlim, ylim=ylim, scan_points=scan_points)
            # plot information gain
            info_list, info_score = informationScore(entropy_map, self.cachedEntropyMap, [1, 0])
            save_info_gain_plot(gain_list=info_list, gain_score=info_score, run_dir=run_dir,
                                cam_points=self.cam_points, placedMarkers=self.placedMarkerPoses, trailMarkers=[],
                                xlim=xlim, ylim=ylim, vminmax=vminmax, scan_points=scan_points)
            winner_entropy_map = None
            for i in range(mark_num):
                best_markID = -1
                high_score = -np.inf
                placedMarkers = self.placedMarkerPoses
                selected_marker_ids, info_gain_percentage = self.most_visible_markers(marker_percentage)
                selected_poses = [self.marker_poses[pose_i] for pose_i in selected_marker_ids]
                save_markers_plot(run_dir, self.cam_points, selected_poses, xlim, ylim, scan_points=scan_points)
                for j in selected_marker_ids:
                    if j not in self.placed_idx:
                        cur_pose = self.marker_poses[j]
                        entropy_map = self.allCamAnalysis(j)
                        self.pickle_res(run_dir, entropy_map, [j])
                        save_locscore_plot(entropy_map, run_dir, cam_points=self.cam_points,
                                           placedMarkers=placedMarkers, trailMarkers=[cur_pose],
                                           xlim=xlim, ylim=ylim, scan_points=scan_points)
                        # plot information gain
                        info_list, info_score, gainAtPercentile = informationScore(entropy_map, self.cachedEntropyMap, [1,0], info_gain_percentage)
                        save_info_gain_plot(gain_list=info_list, gain_score=info_score, run_dir=run_dir,
                                            cam_points=self.cam_points, placedMarkers=placedMarkers,
                                            trailMarkers=[cur_pose],
                                            xlim=xlim, ylim=ylim, vminmax=vminmax,
                                            percentileAndGain=[info_gain_percentage, gainAtPercentile], scan_points=scan_points)
                        info_score = gainAtPercentile
                        if high_score < info_score:
                            high_score = info_score
                            best_markID = j
                            winner_entropy_map = entropy_map
                self.placeMarker(best_markID, winner_entropy_map)
                self.pickle_res(run_dir, winner_entropy_map, [])
                save_locscore_plot(winner_entropy_map, run_dir, cam_points=self.cam_points,
                                   placedMarkers=self.placedMarkerPoses, trailMarkers=[],
                                   xlim=xlim, ylim=ylim, scan_points=scan_points)
                # plot information gain
                info_list, info_score = informationScore(winner_entropy_map, self.cachedEntropyMap, [1,0])
                save_info_gain_plot(gain_list=info_list, gain_score=info_score, run_dir=run_dir,
                                    cam_points=self.cam_points, placedMarkers=self.placedMarkerPoses,
                                    trailMarkers=[],
                                    xlim=xlim, ylim=ylim, vminmax=vminmax, scan_points=scan_points)

                print(f"Placed marker {best_markID}: ", self.marker_poses[best_markID].translation)
                placed_id_file = os.path.join(run_dir, f"current_placement.txt")
                with open(placed_id_file, "w+") as steeam:
                    outstr = " ".join(['ID', 'x', 'y', 'theta'])
                    for id in self.placed_idx:
                        pose = marker_poses[int(id)]
                        x, y, theta = Pose3OnXY(pose)
                        outstr += "\n" + " ".join([str(id), str(x), str(y), str(theta)])
                    steeam.write(outstr)
        else:
            mean_max_weights = np.array(params.mean_max_weights) / np.sum(params.mean_max_weights)
            # save params
            run_dir = self.create_run_dir(save_dir)
            write_params(run_dir, params)
            entropy_map = self.allCamAnalysis()
            self.placeMarker(None, entropy_map)
            self.pickle_res(run_dir, entropy_map, [])

            # plot mean entropy
            save_locscore_plot(entropy_map, run_dir, cam_points=self.cam_points,
                               placedMarkers=self.placedMarkerPoses, trailMarkers=[],
                               xlim=xlim, ylim=ylim, scan_points=scan_points)
            # plot information gain
            info_list, info_score = informationScore(entropy_map, self.cachedEntropyMap, mean_max_weights)
            save_info_gain_plot(gain_list=info_list, gain_score=info_score, run_dir=run_dir,
                                cam_points=self.cam_points, placedMarkers=self.placedMarkerPoses, trailMarkers=[],
                                xlim=xlim, ylim=ylim, vminmax=vminmax, scan_points=scan_points)
            winner_entropy_map = None
            for i in range(mark_num):
                best_markID = -1
                high_score = -np.inf
                placedMarkers = self.placedMarkerPoses
                for j in range(len(self.marker_poses)):
                    if j not in self.placed_idx:
                        cur_pose = self.marker_poses[j]
                        entropy_map = self.allCamAnalysis(j)
                        self.pickle_res(run_dir, entropy_map, [j])
                        save_locscore_plot(entropy_map, run_dir, cam_points=self.cam_points,
                                           placedMarkers=placedMarkers, trailMarkers=[cur_pose],
                                           xlim=xlim, ylim=ylim, scan_points=scan_points)
                        # plot information gain
                        info_list, info_score = informationScore(entropy_map, self.cachedEntropyMap, mean_max_weights)
                        save_info_gain_plot(gain_list=info_list, gain_score=info_score, run_dir=run_dir,
                                            cam_points=self.cam_points, placedMarkers=placedMarkers,
                                            trailMarkers=[cur_pose],
                                            xlim=xlim, ylim=ylim, vminmax=vminmax, scan_points=scan_points)
                        if high_score < info_score:
                            high_score = info_score
                            best_markID = j
                            winner_entropy_map = entropy_map
                self.placeMarker(best_markID, winner_entropy_map)
                self.pickle_res(run_dir, winner_entropy_map, [])
                save_locscore_plot(winner_entropy_map, run_dir, cam_points=self.cam_points,
                                   placedMarkers=self.placedMarkerPoses, trailMarkers=[],
                                   xlim=xlim, ylim=ylim, scan_points=scan_points)
                # plot information gain
                info_list, info_score = informationScore(winner_entropy_map, self.cachedEntropyMap, mean_max_weights)
                save_info_gain_plot(gain_list=info_list, gain_score=info_score, run_dir=run_dir,
                                    cam_points=self.cam_points, placedMarkers=self.placedMarkerPoses,
                                    trailMarkers=[],
                                    xlim=xlim, ylim=ylim, vminmax=vminmax, scan_points=scan_points)

                print(f"Placed marker {best_markID}: ", self.marker_poses[best_markID].translation)
                placed_id_file = os.path.join(run_dir, f"current_placement.txt")
                with open(placed_id_file, "w+") as steeam:
                    outstr = " ".join(['ID', 'x', 'y', 'theta'])
                    for id in self.placed_idx:
                        pose = marker_poses[int(id)]
                        x, y, theta = Pose3OnXY(pose)
                        outstr += "\n" + " ".join([str(id), str(x), str(y), str(theta)])
                    steeam.write(outstr)


def save_fig(fig, save_dir, prefix="fig", dpi=800):
    cnt = 0
    dir = os.path.join(save_dir, f"{prefix}{cnt}.png")
    while os.path.exists(dir):
        cnt += 1
        dir = os.path.join(save_dir, f"{prefix}{cnt}.png")
    fig.savefig(dir, dpi=dpi, bbox_inches='tight')


def extract_marker_poses(data: List[KpScanData], wall_color, sampling_ratio=.05,
                         voxel_size=.02, max_depth=30, min_distance=.2, marker_outlier_distance=.5, save_fig_path=None,
                         use_data_seg=True, use_data_nml=True, save_scan_path=None):
    pcd = o3d.geometry.PointCloud()
    pcd_points = []
    pcd_colors = []
    pcd_nmls = []
    cam_pts = []
    for d in data:
        cam_pts.append(d.Poses[0].translation)
        pts = np.vstack(d.Scans)
        pcd_points.append(pts)
        if use_data_seg:
            cs = np.vstack(d.ScanSegs)[:, :3]
            pcd_colors.append(cs)
        if use_data_nml:
            nml = np.vstack(d.ScanNmls)
            pcd_nmls.append(nml)
    cam_pts = np.array(cam_pts)
    pcd_points = np.vstack(pcd_points)
    pcd_points = np.hstack([pcd_points[:, :2], np.zeros((len(pcd_points), 1))])
    inliers = (np.linalg.norm(pcd_points[:, :2], axis=1) < max_depth)
    pcd.points = o3d.utility.Vector3dVector(pcd_points[inliers])
    if use_data_seg:
        pcd_colors = np.vstack(pcd_colors)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors[inliers])
    if use_data_nml:
        pcd_nmls = np.vstack(pcd_nmls)
        pcd.normals = o3d.utility.Vector3dVector(pcd_nmls[inliers])
    # get wall points
    wall_idx = np.zeros(len(pcd.points), dtype=bool)
    if wall_color is not None and use_data_seg:
        print("Selecting wall voxels using AirSim segmentation")
        dpcd_colors = np.asarray(pcd.colors)
        wall_idx = wall_idx | (np.linalg.norm(dpcd_colors - wall_color, axis=1) < .1)
    else:
        print("Admitting all scan points as wall voxels")
        wall_idx = ~wall_idx
    wall_pcd = o3d.geometry.PointCloud()
    wall_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[wall_idx])
    if use_data_seg:
        wall_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[wall_idx])
    if use_data_nml:
        wall_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[wall_idx])

    # print("Downsample the point cloud with a voxel size")
    downpcd_vox = wall_pcd.voxel_down_sample(voxel_size=voxel_size)
    vox_points = np.asarray(downpcd_vox.points)
    fig, ax = plt.subplots()
    ax.scatter(vox_points[:, 0], vox_points[:, 1], marker=".", c="red", s=.05)
    if save_scan_path is not None:
        np.savetxt(fname=save_scan_path, X=vox_points[:, :2])

    if not use_data_nml:
        print("Estimating surface normals.")
        dz_list = np.array([-.3, -.2, -.1, .1, .2, .3]) * .5
        aug_points = [vox_points]
        for dz in dz_list:
            tmp = vox_points.copy()
            tmp[:, 2] += dz
            aug_points.append(tmp)
        aug_wall_pcd = o3d.geometry.PointCloud()
        aug_wall_pcd.points = o3d.utility.Vector3dVector(np.vstack(aug_points))
        aug_wall_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        zero_plan_idx = (abs(np.asarray(aug_wall_pcd.points)[:, 2]) < .01)
        marker_idx = np.random.choice(np.arange(len(aug_wall_pcd.points))[zero_plan_idx],
                                      size=int(sampling_ratio * len(zero_plan_idx)),
                                      replace=False)
        marker_normals = np.asarray(aug_wall_pcd.normals)[marker_idx]
        marker_points = np.asarray(aug_wall_pcd.points)[marker_idx]
        # will be used for correcting (flipping) normal directions
        arrow_lens = [.1, .2, .3]
    else:
        print("Using surface normals from AirSim.")
        downpcd = downpcd_vox.random_down_sample(sampling_ratio=sampling_ratio)
        # downpcd = downpcd_vox.uniform_down_sample(every_k_points=20)

        marker_normals = np.asarray(downpcd.normals)
        marker_points = np.asarray(downpcd.points)
        # will be used for correcting (flipping) normal directions
        arrow_lens = []  # np.array([1.0])

    # filtering marker points by min distnace
    admit_idx = []
    for i in range(len(marker_points)):
        if len(admit_idx) == 0:
            admit_idx.append(i)
        else:
            tmp_distances = np.linalg.norm(marker_points[i] - marker_points[admit_idx], axis=1)
            tmp_cos_distances = sci_dist.cdist(marker_normals[i:i + 1, :2], marker_normals[admit_idx, :2],
                                               metric="cosine")

            comb_distances = np.sqrt(tmp_distances ** 2 + tmp_cos_distances ** 2)

            if np.min(comb_distances) > min_distance:
                admit_idx.append(i)

    # remove markers whose orientations are pointing outside walls
    new_admit_idx = []
    for i in admit_idx:
        x, y = marker_points[i, :2]
        dx, dy = marker_normals[i, :2]
        arr_head = np.array([x + dx * marker_outlier_distance, y + dy * marker_outlier_distance])
        if np.min(np.linalg.norm(arr_head - cam_pts[:, :2], axis=1)) < marker_outlier_distance:
            new_admit_idx.append(i)

    admit_idx = new_admit_idx

    marker_normals = marker_normals[admit_idx]
    marker_points = marker_points[admit_idx]

    marker_poses = []
    for i in range(len(marker_normals)):
        x, y = marker_points[i, :2]
        dx, dy = marker_normals[i, :2]
        c_votes = 0
        for l in arrow_lens:
            min_dist1 = min(np.linalg.norm(cam_pts[:, :2] - np.array([x + dx * l, y + dy * l]), axis=1))
            min_dist2 = min(np.linalg.norm(cam_pts[:, :2] - np.array([x - dx * l, y - dy * l]), axis=1))
            if min_dist2 < min_dist1:
                c_votes += 1
            else:
                c_votes -= 1
        if c_votes > 0:
            marker_normals[i] *= -1
            dx, dy = marker_normals[i, :2]

        tmp_len = np.sqrt(dx ** 2 + dy ** 2)
        dx, dy = dx / tmp_len, dy / tmp_len
        ax.arrow(x, y, dx * .45, dy * .45, color="black", shape='full', lw=0.8, length_includes_head=True,
                 head_width=.2)
        mat = np.eye(4)
        mat[:3, 3] = marker_points[i]
        mat[:3, 0] = marker_normals[i]
        # we assume the rotation is about the z axis
        mat[0, 1] = -mat[1, 0]
        mat[1, 1] = mat[0, 0]
        marker_poses.append(Pose3(mat))
    ax.scatter(cam_pts[:, 0], cam_pts[:, 1], c="blue", s=.3)
    ax.set_aspect('equal')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if save_fig_path is not None:
        fig.savefig(fname=save_fig_path, bbox_inches="tight", dpi=800)
    return marker_poses


def compute_placement(all_cam_dots: List[KpScanData], marker_poses: List[Pose3], sim_feat_count: List[List[List]],
                      params: ParamStruct, scan_points:np.ndarray):
    mark_std = np.array(params.marker_prior_std)
    T_std = np.array(params.marker_meas_std)
    cam_std = np.array(params.cam_prior_std)
    lmk_std = np.array(params.lmk_prior_std)
    pix_std = params.pix_std

    max_dist = params.imp_max_dist
    max_angle = params.imp_max_angle * np.pi / 180

    w, h, fov = params.img_w, params.img_h, params.fov
    focal_len = w / 2 / np.tan(np.pi * fov / 360)

    placed_id = []

    if os.path.exists(params.placed_marker_path):
        placed_markers = np.loadtxt(params.placed_marker_path, dtype=str)  # ID x y theta
        if len(placed_markers) > 1:
            placed_id += list(placed_markers[1:, 0].astype(int))
    elif os.path.exists(params.imp_pkl_res_path):
        with open(params.imp_pkl_res_path, "rb") as stream:
            pkl_res = pickle.load(stream)
            placed_id += list(pkl_res.PlacedIDs)

    save_dir = os.path.join(params.data_dir, "imp_res")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if len(params.marker_percentages) > 0:
        for m_pctg in params.marker_percentages:
            cur_save_dir = os.path.join(save_dir, f"{m_pctg}percent_markers")
            case = MarkerPlacement(all_cam_dots, marker_poses, lmk_std, mark_std, pix_std, T_std, cam_std,
                                   scale_noise=False, max_dist=max_dist, max_angle=max_angle,
                                   focal_len=focal_len, image_w=w, image_h=h,
                                   sim_feat_count=sim_feat_count, placed_id=placed_id,
                                   ignore_features=params.imp_ignore_features, scan_points=scan_points)
            if not os.path.exists(cur_save_dir):
                os.mkdir(cur_save_dir)
            case.start_placing(params.imp_marker_num, cur_save_dir, params, marker_percentage=m_pctg)
    else:
        case = MarkerPlacement(all_cam_dots, marker_poses, lmk_std, mark_std, pix_std, T_std, cam_std,
                               scale_noise=False, max_dist=max_dist, max_angle=max_angle,
                               focal_len=focal_len, image_w=w, image_h=h,
                               sim_feat_count=sim_feat_count, placed_id=placed_id,
                               ignore_features=params.imp_ignore_features, scan_points=scan_points)
        case.start_placing(params.imp_marker_num, save_dir, params)


def checkCorrectFreeDotsData(pkl_data: List[KpScanData], params: ParamStruct):
    corrcted_data = False
    num_angles = params.fine_angular_resolution
    assert num_angles == len(pkl_data[0].Poses)
    # default tf
    pose0 = Pose3(params.tf_bodyInW)
    folder = os.path.join(params.data_dir, "free_dots_scan_images")
    f_path = os.path.join(folder, "airsim_rec.txt")
    img_folder = os.path.join(folder, "images")
    all_cam_dots = []
    data = np.loadtxt(f_path, delimiter='\t', dtype=str)

    # image settings
    w, h, fov = params.img_w, params.img_h, params.fov
    dep_w, dep_h = params.dep_w, params.dep_h
    focal_len = w / 2 / np.tan(np.pi * fov / 360)
    dep_rgb_scale = dep_w / w

    # scan data
    depth_max = params.depth_max

    # depth type
    dep_type = "Planar"

    # create SIFT
    print(cv.__version__)
    # sift = cv.SIFT.create(nfeatures=100, contrastThreshold=.1, edgeThreshold=5)
    sift = cv.SIFT.create(nfeatures=params.preprocess_nfeature,
                          contrastThreshold=params.preprocess_contrastTh,
                          edgeThreshold=params.preprocess_edgeTh)
    print("Check the pickled data.")
    for dot_id, d in tqdm(enumerate(pkl_data)):
        if (len(d.Keypoints) < len(d.Poses)):
            print(f"Dot {dot_id} only has {len(d.Keypoints)} entries in the keypoint list (expect {len(d.Poses)})")
            kp_list = []
            des_list = []
            min_dist = np.inf
            for angle_id in range(num_angles):
                row_id = dot_id * num_angles + angle_id
                step = row_id + 1
                row = data[step]
                # VehicleName	TimeStamp	POS_X	POS_Y	POS_Z	Q_W	Q_X	Q_Y	Q_Z	ImageFile
                T = [float(row[i]) for i in range(2, 9)]
                cur_pose = pose0 * Pose3.by_vec(*T)
                # process images
                paths = row[9].split(";")
                rgb_name, dep_name, seg_name, sur_name = parseImageName(paths)

                # depth image
                dep_img = read_pfm(os.path.join(img_folder, dep_name))
                # key points
                rgb_img = cv.imread(os.path.join(img_folder, rgb_name))
                gray = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY)
                kp, des = sift.detectAndCompute(gray, None)
                cached_kps = False
                if (len(kp) > 0) & (des is not None):
                    px_z_y = np.array([[round(kp[i].pt[1]), round(kp[i].pt[0])] for i in range(len(kp))])
                    px_d = depth_interpolation(dep_img, px_z_y, dep_rgb_scale)
                    ptInFov = (px_d < depth_max)
                    kp = [kp[i] for i in range(len(kp)) if ptInFov[i]]
                    if len(kp) > 0:
                        des = des[ptInFov]
                        px_z_y = px_z_y[ptInFov]
                        px_d = px_d[ptInFov]
                        body_xyz = px2coor(w, h, px_z_y[:, 1], px_z_y[:, 0], px_d, focal_len, dep_type)
                        body_xyz = np.hstack((body_xyz, np.ones((body_xyz.shape[0], 1))))
                        pt_w_xyz = (cur_pose.mat @ body_xyz.T).T[:, :3]
                        if len(pt_w_xyz.shape) == 1:
                            pt_w_xyz = pt_w_xyz.reshape((1, -1))
                        kp_list.append(pt_w_xyz)
                        if len(des.shape) == 1:
                            des = des.reshape((1, -1))
                        des_list.append(des)
                        cached_kps = True
                        # compare with old data
                        if pkl_data[dot_id].Keypoints[0].shape == pt_w_xyz.shape:
                            min_dist = min(min_dist, np.linalg.norm(pkl_data[dot_id].Keypoints[0] - pt_w_xyz))
                if not cached_kps:
                    print(f"No key points for step {row_id}.")
                    kp_list.append(None)
                    des_list.append(None)
            pkl_data[dot_id].Descriptors = des_list
            pkl_data[dot_id].Keypoints = kp_list
            corrcted_data = True
            if min_dist > .1:
                raise ValueError("Wrong correction.")
    return corrcted_data, pkl_data


def preprocess(params: ParamStruct):
    num_angles = params.fine_angular_resolution
    folder = os.path.join(params.data_dir, "free_dots_scan_images")
    img_folder = os.path.join(folder, "images")
    f_path = os.path.join(folder, "airsim_rec.txt")
    all_cam_dots = []
    data = np.loadtxt(f_path, delimiter='\t', dtype=str)
    # print(data[0])

    overwrite_data = params.overwrite_data

    # default tf
    pose0 = Pose3(params.tf_bodyInW)

    # setting
    pose_list, scan_list, kp_list, seg_list, nml_list, des_list, = [], [], [], [], [], []

    # load data
    pickle_file = os.path.join(params.data_dir, 'free_dots_data.pkl')
    if os.path.exists(pickle_file) and not overwrite_data:
        with open(pickle_file, "rb") as f:
            all_cam_dots = pickle.load(f)
        corrected, all_cam_dots = checkCorrectFreeDotsData(all_cam_dots, params)
        if corrected:
            with open(pickle_file, "wb") as f:
                pickle.dump(all_cam_dots, f)
    else:
        # save data
        save_feature_txt = params.save_feature_txt
        save_feature_img = params.save_feature_img
        save_mid_scan = params.save_scan_txt
        scan_folder = None
        sift_folder = None
        if save_mid_scan:
            scan_folder = os.path.join(folder, "scan")
            if not os.path.exists(scan_folder):
                os.mkdir(scan_folder)
        if save_feature_txt or save_feature_img:
            sift_folder = os.path.join(folder, "feature")
            if not os.path.exists(sift_folder):
                os.mkdir(sift_folder)
        # image settings
        w, h, fov = params.img_w, params.img_h, params.fov
        dep_w, dep_h = params.dep_w, params.dep_h
        focal_len = w / 2 / np.tan(np.pi * fov / 360)
        dep_rgb_scale = dep_w / w
        mid_r = int(h / 2)
        px_y = np.arange(0, w)
        px_z = np.ones(w, dtype=int) * int(h / 2)

        # scan data
        depth_max = params.depth_max

        # depth type
        dep_type = "Planar"

        # create SIFT
        print(cv.__version__)
        # sift = cv.SIFT.create(nfeatures=100, contrastThreshold=.1, edgeThreshold=5)
        sift = cv.SIFT.create(nfeatures=params.preprocess_nfeature,
                              contrastThreshold=params.preprocess_contrastTh,
                              edgeThreshold=params.preprocess_edgeTh)

        print("Preprocessing free dots data...")
        for step in tqdm(range(1, len(data))):
            row_id = step - 1
            row = data[step]
            # VehicleName	TimeStamp	POS_X	POS_Y	POS_Z	Q_W	Q_X	Q_Y	Q_Z	ImageFile
            T = [float(row[i]) for i in range(2, 9)]
            cur_pose = pose0 * Pose3.by_vec(*T)
            pose_list.append(cur_pose)

            # process images
            paths = row[9].split(";")
            rgb_name, dep_name, seg_name, sur_name = parseImageName(paths)

            # depth image
            dep_img = read_pfm(os.path.join(img_folder, dep_name))
            mid_rgb_r_pix = np.ones((w, 2))
            mid_rgb_r_pix[:, 0] = px_z
            mid_rgb_r_pix[:, 1] = px_y
            px_d = depth_interpolation(dep_img, mid_rgb_r_pix, dep_rgb_scale)
            body_xyz = px2coor(w, h, px_y, px_z, px_d, focal_len, dep_type)
            body_xyz = np.hstack((body_xyz, np.ones((body_xyz.shape[0], 1))))
            # beam for occupancy grid mapping
            w_xyz = (cur_pose.mat @ body_xyz.T).T[:, :3]

            # seg data
            if seg_name is not None:
                seg_img = Image.open(os.path.join(img_folder, seg_name))  # Can be many different formats.
                seg_pix = seg_img.load()
                mid_seg = np.array([seg_pix[i, mid_r] for i in px_y]) / 255.0001
            else:
                mid_seg = None

            if sur_name is not None:
                nml_img = Image.open(os.path.join(img_folder, sur_name))
                nml_pix = nml_img.load()
                nml_vec = np.array([nml_pix[i, mid_r] for i in px_y]) / 255.0 * 2.0 - 1.0
                mid_nml = (pose0.rotation @ nml_vec[:, :3].T).T
            else:
                mid_nml = None

            if save_mid_scan:
                np.savetxt(fname=os.path.join(scan_folder, f"scan{row_id}.txt"), X=w_xyz)
                if mid_seg is not None:
                    np.savetxt(fname=os.path.join(scan_folder, f"seg{row_id}.txt"), X=mid_seg)
                if mid_nml is not None:
                    np.savetxt(fname=os.path.join(scan_folder, f"nml{row_id}.txt"), X=mid_nml)

            # scan
            scan_list.append(w_xyz)
            if mid_seg is not None:
                seg_list.append(mid_seg)
            if mid_nml is not None:
                nml_list.append(mid_nml)

            # key points
            rgb_img = cv.imread(os.path.join(img_folder, rgb_name))
            gray = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            cached_kps = False
            if (len(kp) > 0) & (des is not None):
                if save_feature_img:
                    rbg_img = cv.drawKeypoints(gray, kp, rgb_img,
                                               flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv.imwrite(os.path.join(sift_folder, f'sift{row_id}.png'), rbg_img)
                px_z_y = np.array([[round(kp[i].pt[1]), round(kp[i].pt[0])] for i in range(len(kp))])
                px_d = depth_interpolation(dep_img, px_z_y, dep_rgb_scale)
                ptInFov = (px_d < depth_max)
                kp = [kp[i] for i in range(len(kp)) if ptInFov[i]]
                if len(kp) > 0:
                    des = des[ptInFov]
                    px_z_y = px_z_y[ptInFov]
                    px_d = px_d[ptInFov]
                    body_xyz = px2coor(w, h, px_z_y[:, 1], px_z_y[:, 0], px_d, focal_len, dep_type)
                    body_xyz = np.hstack((body_xyz, np.ones((body_xyz.shape[0], 1))))
                    pt_w_xyz = (cur_pose.mat @ body_xyz.T).T[:, :3]
                    raw_px_z_y = np.array([[pt.pt[0], pt.pt[1]] for pt in kp])
                    if save_feature_txt:
                        saved_kp = np.hstack([pt_w_xyz, raw_px_z_y.reshape((len(kp), -1))])
                        np.savetxt(fname=os.path.join(sift_folder, f"kp{row_id}.txt"), X=saved_kp)
                        np.savetxt(fname=os.path.join(sift_folder, f"des{row_id}.txt"), X=des)
                    if len(pt_w_xyz.shape) == 1:
                        pt_w_xyz = pt_w_xyz.reshape((1, -1))
                    kp_list.append(pt_w_xyz)
                    if len(des.shape) == 1:
                        des = des.reshape((1, -1))
                    des_list.append(des)
                    cached_kps = True
            if not cached_kps:
                print(f"No key points for step {row_id}.")
                kp_list.append(None)
                des_list.append(None)
            if step % num_angles == 0:
                all_cam_dots.append(KpScanData(pose_list, kp_list, scan_list, seg_list, nml_list, des_list))
                pose_list, scan_list, kp_list, seg_list, nml_list, des_list = [], [], [], [], [], []
        with open(pickle_file, "wb") as f:
            pickle.dump(all_cam_dots, f)

    # load marker data
    pickle_file = os.path.join(params.data_dir, 'candidate_marker_poses.pkl')
    if os.path.exists(pickle_file) and not overwrite_data:
        with open(pickle_file, "rb") as f:
            marker_poses = pickle.load(f)
    else:
        use_airsim_seg = True
        use_airsim_nml = True
        if len(all_cam_dots[0].ScanSegs) < len(all_cam_dots[0].Scans):
            print("No segmentation data from AirSim. Admitting all points in the scan as walls.")
            use_airsim_seg = False
        if len(all_cam_dots[0].ScanNmls) < len(all_cam_dots[0].Scans):
            print("No surface normal data from AirSim. Estimating surface normals.")
            use_airsim_nml = False
        wall_color = np.array(params.wall_color) / 255.0001
        marker_poses = extract_marker_poses(all_cam_dots, wall_color,
                                            sampling_ratio=params.marker_sampling_ratio,
                                            voxel_size=params.marker_voxel_size,
                                            max_depth=params.depth_max, use_data_seg=use_airsim_seg,
                                            use_data_nml=use_airsim_nml,
                                            min_distance=params.marker_min_distance,
                                            marker_outlier_distance=params.marker_outlier_distance,
                                            save_fig_path=os.path.join(params.data_dir, "candidate_markers.png"),
                                            save_scan_path=os.path.join(params.data_dir, "scan_point_map.txt"))
        with open(pickle_file, "wb") as f:
            pickle.dump(marker_poses, f)
    print("number of candidate markers: ", len(marker_poses))
    # load marker data
    pickle_file = os.path.join(params.data_dir, 'candidate_marker_poses.txt')
    if not os.path.exists(pickle_file) or overwrite_data:
        with open(pickle_file, "w") as f:
            outstr = " ".join(["ID", "POS_X", "POS_Y", "POS_Z", "Q_W", "Q_X", "Q_Y", "Q_Z"])
            for i, pose in enumerate(marker_poses):
                tmp_str = [str(i)] + [str(s) for s in pose.transQuat]
                outstr = outstr + "\n" + " ".join(tmp_str)
            f.write(outstr)
    return all_cam_dots, marker_poses


def post_process(free_dots_data_dir, plot_dir, mean_max_weights=[.5, .5], xlim=[-15, 15], ylim=[-15, 25], vminmax=[]):
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    cnt = 0
    pickle_file = os.path.join(free_dots_data_dir, f"res{cnt}.pkl")
    lowest_score = np.inf
    winner_entropy_list = None
    cam_points = None
    winner_markers = None
    while os.path.exists(pickle_file):
        # load data
        with open(pickle_file, "rb") as f:
            res = pickle.load(f)  # load ResultCase
        cam_points, placedMarkers, trail, entropy_map = res.Points, res.PlacedMarkers, res.TrialMarkers, res.EntropyMap
        entropy_list = entropyScore(entropy_map, mean_max_weights)
        fig, ax = plt.subplots(figsize=(6, 8))
        fig, ax = visualize(fig, ax, cam_points, entropy_list, placedMarkers, trail, vminmax, 'binary')
        h_score = entropyScore(entropy_list, mean_max_weights)
        ax.set_title(
            f"Score: {round(h_score, 4)}, Weight of Mean: {round(mean_max_weights[0], 2)}")
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_aspect('equal', 'box')
        save_fig(fig, plot_dir)
        if lowest_score > h_score:
            lowest_score = h_score
            winner_entropy_list = entropy_list
            winner_markers = placedMarkers + trail
        cnt += 1
        pickle_file = os.path.join(free_dots_data_dir, f"res{cnt}.pkl")

    fig, ax = plt.subplots(figsize=(6, 8))
    fig, ax = visualize(fig, ax, cam_points, winner_entropy_list, winner_markers, [], vminmax, 'binary')
    h_score = entropyScore(winner_entropy_list, mean_max_weights)
    ax.set_title(
        f"Score: {round(h_score, 4)}, Weight of Mean: {round(mean_max_weights[0], 2)}")
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_aspect('equal', 'box')
    save_fig(fig, plot_dir)


def compute_feature_sim(step, query_pts, query_des, train_points, train_descriptors, min_sim_score, min_distance,
                        batch_size):
    # distances
    print(f"Counting similar features for step {step}")
    step_matched_cnts = []
    idx0, idx1 = 0, len(query_pts)
    init_idx = idx0
    end_idx = batch_size + idx0
    while init_idx < idx1:
        if end_idx > idx1:
            end_idx = idx1
        # distances
        dist_mat = sci_dist.cdist(query_pts[init_idx:end_idx], train_points)
        sim_mat = cosine_similarity(query_des[init_idx:end_idx], train_descriptors)
        # matched distant features
        matched_mat = (sim_mat > min_sim_score) & (dist_mat > min_distance)
        tmp_matched_cnt = np.sum(matched_mat, axis=1)
        step_matched_cnts = np.concatenate((step_matched_cnts, tmp_matched_cnt))
        init_idx = end_idx
        end_idx = init_idx + batch_size
    return step_matched_cnts


def feature_similarity(data: List[KpScanData], min_sim_score: float = .9, min_distance: float = 2.0,
                       parallel_jobs=8, batch_size=100, show_pcl=False):
    num_angles = len(data[0].Poses)
    num_steps = len(data)
    points_list = []
    descriptor_list = []
    angle_id_list = []
    step2idx = [0]
    print("Reading keypoints and descriptor data.")
    for step in tqdm(range(num_steps)):
        valid_angle_ids = [i for i in range(num_angles) if data[step].Keypoints[i] is not None]
        cur_points = []
        cur_angle_idx = []
        cur_pos_idx = []
        cur_descriptors = []
        for angle_idx in valid_angle_ids:
            cur_points.append(data[step].Keypoints[angle_idx])
            cur_angle_idx.append([angle_idx] * len(cur_points[-1]))
            cur_pos_idx.append([step] * len(cur_points[-1]))
            cur_descriptors.append(data[step].Descriptors[angle_idx])
        points_list.append(np.vstack(cur_points))
        descriptor_list.append(np.vstack(cur_descriptors))
        angle_id_list.append(np.hstack(cur_angle_idx))
        step2idx.append(step2idx[step] + len(points_list[-1]))
    points_list = np.vstack(points_list)
    descriptor_list = np.vstack(descriptor_list)

    # downsample the points for counting similar features
    # create point clouds using feature points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_list)
    axis_aligned_bounding_box = pcd.get_axis_aligned_bounding_box()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    ds_pcd, idx_map, old_pcd_idx = pcd.voxel_down_sample_and_trace(voxel_size=params.feat_voxel_size,
                                                                   min_bound=axis_aligned_bounding_box.min_bound,
                                                                   max_bound=axis_aligned_bounding_box.max_bound)
    lens = [len(idx) for idx in old_pcd_idx]
    colors = np.array(lens) / (max(lens) + .001)
    colors = np.hstack([colors.reshape((-1, 1)), np.zeros((len(colors), 1)), np.zeros((len(colors), 1))])
    ds_pcd.colors = o3d.utility.Vector3dVector(colors)
    if show_pcl:
        o3d.visualization.draw(
            [ds_pcd, axis_aligned_bounding_box, mesh_frame])
    ds_descriptor_list = [np.mean(descriptor_list[idx], axis=0) for idx in old_pcd_idx]
    ds_points_list = np.asarray(ds_pcd.points)

    # t = time.time()
    if parallel_jobs > 1:
        feature_matches = Parallel(n_jobs=parallel_jobs)(
            delayed(compute_feature_sim)(step, points_list[step2idx[step]:step2idx[step + 1]],
                                         descriptor_list[step2idx[step]:step2idx[step + 1]],
                                         ds_points_list, ds_descriptor_list,
                                         min_sim_score, min_distance, batch_size) for step in range(num_steps))
    else:
        feature_matches = []
        for step in tqdm(range(num_steps)):
            step_matched_cnts = []
            idx0, idx1 = step2idx[step], step2idx[step + 1]
            init_idx = idx0
            end_idx = batch_size + idx0
            while init_idx < idx1:
                # print(f"Counting for feature {init_idx} out of {step2idx[-1]}")
                if end_idx > idx1:
                    end_idx = idx1
                query_pts = points_list[init_idx:end_idx]
                query_des = descriptor_list[init_idx:end_idx]
                # distances
                dist_mat = sci_dist.cdist(query_pts, ds_points_list)
                sim_mat = cosine_similarity(query_des, ds_descriptor_list)
                # matched distant features
                matched_mat = (sim_mat > min_sim_score) & (dist_mat > min_distance)
                tmp_matched_cnt = np.sum(matched_mat, axis=1)
                step_matched_cnts = np.concatenate((step_matched_cnts, tmp_matched_cnt))
                init_idx = end_idx
                end_idx = init_idx + batch_size
            feature_matches.append(step_matched_cnts)

    # print(time.time() - t)

    # num of similar features
    num_sim_features = [[[]] * num_angles for _ in range(num_steps)]
    for step in tqdm(range(num_steps)):
        step_angle_ids = angle_id_list[step]
        step_feature_cnts = feature_matches[step]
        assert len(step_angle_ids) == len(step_feature_cnts)
        for angle_idx in range(num_angles):
            feat_idx = np.where(step_angle_ids == angle_idx)[0]
            count_featuers = [step_feature_cnts[i] for i in feat_idx]
            num_sim_features[step][angle_idx] = count_featuers
    return num_sim_features

if __name__ == "__main__":
    params = imp_setting("omp_settings.yaml")
    np.random.seed(params.random_seed)
    random.seed(params.random_seed)
    all_cam_dot_data, marker_poses = preprocess(params)

    pickle_file = os.path.join(params.data_dir, 'similar_feat_cnt.pkl')
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            step_angle_feat_num = pickle.load(f)
    else:
        step_angle_feat_num = feature_similarity(all_cam_dot_data,
                                                 min_sim_score=params.feat_min_sim_score,
                                                 min_distance=params.feat_min_distance,
                                                 parallel_jobs=params.feat_parallel_jobs,
                                                 batch_size=params.feat_batch_size)
        with open(pickle_file, "wb") as f:
            pickle.dump(step_angle_feat_num, f)
    scan_points = np.loadtxt(os.path.join(params.data_dir, "scan_point_map.txt"), dtype=float)
    compute_placement(all_cam_dot_data, marker_poses, step_angle_feat_num, params, scan_points)
