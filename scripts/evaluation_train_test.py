import gc
from copy import deepcopy

import sklearn
from dt_apriltags import Detector, Detection
from typing import Union, List
from dataclasses import dataclass

from scipy.stats import circmean
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from joblib import Parallel, delayed
from matplotlib import markers, path

from evaluation_preprocess import TrainData, TestData, get_test_data, get_train_data, get_id_vlad_des
from utils import Pose3OnXY, imp_setting, Pose3

def FlannGoodMatches(test_des: List[np.ndarray], train_des: List[np.ndarray], flann: cv.FlannBasedMatcher, k=2,
                     ratio=.8):
    all_matches = []
    for test_i, des1, in enumerate(test_des):
        for train_j, des2 in enumerate(train_des):
            good_match = []
            matches = flann.knnMatch(des1, des2, k=k)
            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < ratio * n.distance:
                    good_match.append(m)


def SingleTestImg(test_img_id, test_cam_pose, tags: List[Detection], des: np.ndarray, kps: List,
                  train_data: TrainData, train_cam_xyt: np.ndarray, numNNs: int, train_des_id: List,
                  train_vlad_des: np.ndarray, test_des_id: List, test_vlad_des: np.ndarray,
                  flann_k=2, ratio_test=.8, rank_tags="vlad", tag_loc_percent=1.0, rd_seed=1, matchedImgID=[], bf_match=True):
    if des is None or len(des) == 0:
        return None
    else:
        if isinstance(des, list):
            des = np.array([des])
        tag_loc_nn = max(0, min(int(numNNs * tag_loc_percent), numNNs))

        tag_matched_train_id = set()
        all_matched_train_id = set()
        if tag_loc_nn>0:
            for tag in tags:
                if tag.tag_id in train_data.TagID2ImgID:
                    tag_matched_train_id.update(train_data.TagID2ImgID[tag.tag_id])

                # get train camera poses that are close to the projected test camera pose
                if tag.tag_id in train_data.TagOrder2Poses:
                    tag_pose = np.eye(4)
                    tag_pose[:3,:3] = tag.pose_R
                    tag_pose[:3, 3] = tag.pose_t.flatten()
                    tag_pose = Pose3(tag_pose)
                    proj_test_cam = train_data.TagOrder2Poses[tag.tag_id] * tag_pose.inverse()
                    proj_test_xyt = Pose3OnXY(proj_test_cam)
                    nearby_train_cams = (np.linalg.norm( train_cam_xyt[:, :2] - proj_test_xyt[:2], axis=1) < params.trans_tol) & \
                                        (abs((train_cam_xyt[:,2]-proj_test_xyt[-1]+np.pi)%(2*np.pi)-np.pi) < params.rad_tol)
                    nearby_train_cam_idx = np.arange(len(train_cam_xyt))[nearby_train_cams]
                    tag_matched_train_id.update(nearby_train_cam_idx)

            if len(tag_matched_train_id) > tag_loc_nn:
                tag_matched_train_id = [img_id for img_id in tag_matched_train_id if img_id in train_des_id]

            if len(tag_matched_train_id) > tag_loc_nn:
                if rank_tags == "feature_num":
                    # pick train images that have more features
                    tag_matched_train_id = list(tag_matched_train_id)
                    des_counts = [len(train_data.Keypoints[i]) for i in tag_matched_train_id]
                    cnt_list, id_list = zip(*sorted((zip(des_counts, tag_matched_train_id))))
                    tag_matched_train_id = id_list[-tag_loc_nn:]  # id of images with more features
                elif rank_tags == "vlad": # pick by VLAD descriptors
                    tag_matched_train_id = list(tag_matched_train_id)
                    tag_vlad = np.array([train_vlad_des[train_des_id.index(img_id)] for img_id in tag_matched_train_id])
                    test_des = test_vlad_des[test_des_id.index(test_img_id)]
                    dist_list = np.linalg.norm(tag_vlad - test_des.reshape(1, -1), axis=1)
                    sort_dist_list, id_list = zip(*sorted((zip(dist_list, tag_matched_train_id))))
                    tag_matched_train_id = id_list[:tag_loc_nn]
                else:
                    raise ValueError(f"Unknown method for ranking images with tags {rank_tags}")
            all_matched_train_id.update(tag_matched_train_id)

        if len(all_matched_train_id) < numNNs:
            # use vlad to find matches in remaining training images
            test_des = test_vlad_des[test_des_id.index(test_img_id)]
            dist_list = np.linalg.norm(train_vlad_des - test_des.reshape(1, -1), axis=1)
            sort_dist_list, id_list = zip(*sorted((zip(dist_list, train_des_id))))
            ind = [i for i in id_list if i not in all_matched_train_id]
            all_matched_train_id.update(ind[:(numNNs-len(all_matched_train_id))])

    matched_train_id = list(all_matched_train_id)
    save_idx = [str(test_img_id)]
    save_idx += [str(t_id) for t_id in matched_train_id]
    matchedImgID.append(save_idx)

    cv.setRNGSeed(rd_seed)
    pts3d = []
    pts2d = []
    test_ids = [tag.tag_id for tag in tags]
    for train_id in matched_train_id:
        train_kp = deepcopy(train_data.Keypoints[train_id])
        train_des = deepcopy(train_data.Descriptors[train_id])
        train_xyz = deepcopy(train_data.PointsXYZ[train_id])

        try:
            if bf_match:
                bf = cv.BFMatcher()
                matches = bf.knnMatch(des, train_des, k=flann_k)
            else:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)  # or pass empty dictionary
                flann = cv.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des, train_des, k=flann_k)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(f"Zero matches: test img id {test_img_id}, train img id: {train_id}.")
            matches = []
        # ratio test as per Lowe's paper
        for i, mnpair in enumerate(matches):
            if len(mnpair) == 2:
                m, n = mnpair
                if m.distance < ratio_test * n.distance:
                    pts3d.append(train_xyz[m.trainIdx])
                    pts2d.append(kps[m.queryIdx])
        if len(matches)>0:
            if len(matches[0]) == 1:
                print(f"No match pair: test img id {test_img_id}, train img id: {train_id}.")
    pts2d = np.array(pts2d)
    pts3d = np.array(pts3d)
    try:
        solved, rvec, tvec, inliers = cv.solvePnPRansac(pts3d, pts2d, train_data.ProjectMat, np.zeros(4),
                                                        iterationsCount=200, reprojectionError=4.0,
                                                        flags=cv.SOLVEPNP_P3P)

        if solved:
            retval, rvec, tvec = cv.solvePnP(pts3d[inliers], pts2d[inliers], train_data.ProjectMat, np.zeros(4),
                                             useExtrinsicGuess=True, rvec=rvec, tvec=tvec, flags=cv.SOLVEPNP_ITERATIVE)

            mat = np.eye(4)
            mat[:3, :3] = cv.Rodrigues(rvec)[0]
            mat[:3, 3] = tvec.flatten()
            pose = Pose3(mat).inverse()
            return pose  # cv returns the world frame in the camera frame
        else:
            return None
    except:
        return None


def MatchImgEstimatePose(test_data: TestData, train_data: TrainData, train_cam_xyt: np.ndarray, numNNs: int,
                         train_des_id, train_vlad_des, test_des_id, test_vlad_des, parallel_jobs=1, tag_loc_percent=.5,
                         rd_seed=1, matchedImgID=[]):
    if parallel_jobs > 1:
        return Parallel(n_jobs=parallel_jobs)(delayed(SingleTestImg)(i, test_data.CamPoses[i],
                                                                     test_data.Tags[i],
                                                                     test_data.Descriptors[i],
                                                                     test_data.Keypoints[i],
                                                                     train_data,train_cam_xyt, numNNs,
                                                                     train_des_id, train_vlad_des, test_des_id, test_vlad_des,
                                                                     tag_loc_percent=tag_loc_percent,
                                                                     rd_seed=rd_seed,
                                                                     matchedImgID=matchedImgID) for i in
                                              range(len(test_data.CamPoses)))
    else:
        res = []
        for i in tqdm(range(len(test_data.CamPoses))):
            res.append(SingleTestImg(i, test_data.CamPoses[i],
                                     test_data.Tags[i],
                                     test_data.Descriptors[i],
                                     test_data.Keypoints[i],
                                     train_data,train_cam_xyt, numNNs,
                                     train_des_id, train_vlad_des, test_des_id, test_vlad_des,
                                     tag_loc_percent=tag_loc_percent,
                                     rd_seed=rd_seed, matchedImgID=matchedImgID))
        return res


def visualize(xs, ys, terr, Rerr,
              placedMarkers: List[Pose3] = [], t_limits=[0, 2], R_limits=[0, np.pi / 2], marker_size=10):
    placed_color = "blue"
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for i, pose in enumerate(placedMarkers):
        arrow = markers.MarkerStyle(marker='$\u2193$')  # Downwards arrow in Unicode: ↓
        x, y, theta = pose.mat[0, 3], pose.mat[1, 3], np.arctan2(pose.mat[1, 2], pose.mat[0, 2])
        arrow._transform = arrow.get_transform().rotate_deg(90 + theta * 180 / np.pi)
        axes[0].scatter((x), (y), marker=arrow, s=100, c=placed_color)
        axes[0].text((x + .5), (y - .5), str(i + 1))
        axes[1].scatter((x), (y), marker=arrow, s=100, c=placed_color)
        axes[1].text((x + .5), (y - .5), str(i + 1))

    res = axes[0].scatter(xs, ys, marker='.', s=marker_size,
                          c=terr, vmin=t_limits[0], vmax=t_limits[1])
    fig.colorbar(res, ax=axes[0])
    res = axes[1].scatter(xs, ys, marker='.', s=marker_size,
                          c=Rerr, vmin=R_limits[0], vmax=R_limits[1])
    fig.colorbar(res, ax=axes[1])

    return fig, axes

if __name__ == "__main__":
    try:
        import cPickle as pickle
    except ModuleNotFoundError:
        import pickle
    params = imp_setting("omp_settings.yaml")

    at_detector = Detector(families=params.tag_family,
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)

    use_nomarker_vlad = False
    eva_folder = os.path.join(params.data_dir, "evaluation")
    no_marker_case = "NoMarker"
    if 0 in params.eva_marker_nums:
        case_names = [no_marker_case]
    else:
        case_names = []
    case_names += [f"Pct{i}Marker" for i in params.marker_percentages]
    case_names += [f"Rd{i}Marker" for i in params.eva_random_case_ids]
    case_names += [f"Uc{i}Marker" for i in params.eva_unif_contour_case_ids]

    bodyInW = Pose3(params.tf_bodyInW)
    camInB = Pose3(params.tf_camInB)

    # VLAD setting
    numVisualWords = params.VLAD_visual_words
    ballTreeLeafSize = params.ball_tree_leaf_size
    # coarse localization
    numNNs = params.nearest_neighbor_imgs

    xlim = [params.grid_x_offset, params.grid_x_offset + params.grid_x_len]
    ylim = [params.grid_y_offset, params.grid_y_offset + params.grid_y_len]

    save_marker_plot = True

    test_cases = []
    for suffix in params.test_data_distributions:
        if suffix == "unif":
            test_cases.append(f"test_{suffix}")
        else:
            for mean_scale in params.test_sample_mean_weight_scales:
                test_cases.append(f"test_{suffix}_meanScale{mean_scale}")

    for c_i, c_name in enumerate(case_names):
        for m_num in params.eva_marker_nums:
            print(f"solving {c_name} with {m_num} markers...")
            if m_num == 0 and c_name == no_marker_case:
                case_folder = os.path.join(eva_folder, c_name)
                marker_name = None
            elif m_num != 0 and c_name != no_marker_case:
                case_folder = os.path.join(eva_folder, c_name, f"{m_num}markers")
                marker_name = f"tag{c_name}"
            else:
                continue

            pkl_folder = f"{case_folder}/pkl"
            if not os.path.exists(pkl_folder):
                os.mkdir(pkl_folder)

            res_folder = os.path.join(pkl_folder, "result")

            if not os.path.exists(res_folder):
                os.mkdir(res_folder)

            if c_name == no_marker_case:
                last_test_case = os.path.join(res_folder, f'test_perf_{test_cases[-1]}_0tagloc.txt')
            else:
                last_test_case = os.path.join(res_folder, f'test_perf_{test_cases[-1]}_{params.tag_loc_percent[-1]}tagloc.txt')
            if os.path.exists(last_test_case):
                continue
            # preparing training data (the map data)
            train_frame_num = None
            test_frame_num = None
            train_data = get_train_data(pkl_folder, case_folder, params, train_frame_num, marker_name, save_marker_plot, at_detector=at_detector)
            train_cam_xyt = np.array([Pose3OnXY(tmp_pose) for tmp_pose in train_data.CamPoses])

            if use_nomarker_vlad:
                train_des_id, train_vlad_des = get_id_vlad_des(train_data.Descriptors, "train", os.path.join(eva_folder, no_marker_case, "pkl"), numVisualWords)
            else:
                train_des_id, train_vlad_des = get_id_vlad_des(train_data.Descriptors, "train", pkl_folder, numVisualWords)

            if c_name == no_marker_case:
                tag_loc_frac = [0]
                suffixes = [f"{frac}tagloc" for frac in tag_loc_frac]
            else:
                tag_loc_frac = params.tag_loc_percent
                suffixes = [f"{frac}tagloc" for frac in tag_loc_frac]
            for test_folder in test_cases:
                test_data = get_test_data(pkl_folder, case_folder, test_folder, params, test_frame_num, at_detector=at_detector)

                if use_nomarker_vlad:
                    test_des_id, test_vlad_des = get_id_vlad_des(test_data.Descriptors, test_folder,
                                                                os.path.join(eva_folder, no_marker_case, "pkl"), numVisualWords)
                else:
                    test_des_id, test_vlad_des = get_id_vlad_des(test_data.Descriptors, test_folder, pkl_folder, numVisualWords)

                est_pkls = [os.path.join(res_folder, f'est_poses_{test_folder}_{suffix}.pkl') for suffix in suffixes]
                matchedIDs = [os.path.join(res_folder, f'matchedID_{test_folder}_{suffix}.txt') for suffix in suffixes]
                est_poses = []

                for est_i, est_pkl in enumerate(est_pkls):
                    # use_tag = use_tags[est_i]
                    if not os.path.exists(est_pkl):
                        matchedImgID = []
                        est_test_cam_poses = MatchImgEstimatePose(test_data, train_data,
                                                                  train_cam_xyt, numNNs,
                                                                  train_des_id, train_vlad_des, test_des_id, test_vlad_des,
                                                                  tag_loc_percent=tag_loc_frac[est_i],
                                                                  rd_seed=params.random_seed, matchedImgID=matchedImgID)
                        with open(os.path.join(matchedIDs[est_i]), "w+") as f:
                            f.write('\n'.join([' '.join(line) for line in matchedImgID]))
                        with open(est_pkl, "wb") as f:
                            pickle.dump(est_test_cam_poses, f, pickle.HIGHEST_PROTOCOL)
                    else:
                        with open(est_pkl, "rb") as f:
                            est_test_cam_poses = pickle.load(f)
                    est_poses.append(est_test_cam_poses)

                    if not os.path.exists(os.path.join(res_folder, f'test_perf_{test_folder}_{suffixes[est_i]}.txt')):
                        x_list = []
                        y_list = []
                        terr_list = []
                        Rerr_list = []
                        img_ids = []
                        for i, pose in tqdm(enumerate(est_test_cam_poses)):
                            if pose is not None:
                                t_err = np.linalg.norm(pose.translation - test_data.CamPoses[i].translation)
                                temp = np.clip((np.trace(pose.rotation.T @ test_data.CamPoses[i].rotation) - 1) / 2, -1,
                                               1)
                                R_err = abs(np.arccos(temp)) * 180 / np.pi
                                # print(f"test img {i} rotation err {R_err} deg translation err {t_err} m")
                                x_list.append(test_data.CamPoses[i].translation[0])
                                y_list.append(test_data.CamPoses[i].translation[1])
                                terr_list.append(t_err)
                                Rerr_list.append((R_err))
                                img_ids.append(i)
                        terr_list = np.array(terr_list)
                        Rerr_list = np.array(Rerr_list)
                        err_arr = np.hstack((terr_list.reshape((-1, 1)), Rerr_list.reshape((-1, 1)),
                                             np.array(img_ids).reshape((-1, 1))))
                        np.savetxt(fname=os.path.join(res_folder, f'test_err_{test_folder}_{suffixes[est_i]}.txt'),
                                   X=err_arr, header="TranslationError(m) RotationError(deg)")

                        trans_angle_max = params.perf_trans_angle_max

                        metric_str = ["LocRate", "TEmedian", "REmedian", "TEmean", "REmean"]
                        metric_str += [f"TE{t_a[0]}RE{t_a[1]}" for i, t_a in enumerate(trans_angle_max)]
                        metric_values = [len(terr_list) / len(est_test_cam_poses),
                                         np.median(terr_list), np.median(Rerr_list),
                                         np.mean(terr_list), circmean(Rerr_list * np.pi / 180) * 180 / np.pi]
                        metric_values += [np.sum((terr_list < t_a[0]) & (Rerr_list < t_a[1])) / len(est_test_cam_poses)
                                          for t_a in trans_angle_max]
                        metric_values = np.array(metric_values)
                        fig, axes = visualize(x_list, y_list, terr_list, Rerr_list, [])
                        fig.savefig(fname=os.path.join(res_folder, f"test_scatter_{test_folder}_{suffixes[est_i]}.png"),
                                    dpi=300, bbox_inches='tight')
                        np.savetxt(fname=os.path.join(res_folder, f'test_perf_{test_folder}_{suffixes[est_i]}.txt'),
                                   X=metric_values.reshape((1, -1)), header=" ".join(metric_str))
                del test_data
                gc.collect()
            del train_data
            gc.collect()
