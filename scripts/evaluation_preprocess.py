import gc
from copy import deepcopy

from dt_apriltags import Detector, Detection
from typing import List
from dataclasses import dataclass

from scipy.stats import circmean
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from joblib import Parallel, delayed
from matplotlib import markers
from utils import Pose3OnXY, imp_setting, depth_interpolation, read_pfm, px2coor, Pose3, ParamStruct
import pickle
from VLAD import kMeansDictionary, VLAD
import msgpack
import time

def plot_frame(ax, pose: Pose3, label=None, length=1):
    x, y, z = pose.translation
    rot = pose.rotation
    colors = ['red', 'green', 'blue']
    if label is not None:
        ax.text(x, y, z, label)
    for i in range(3):
        ax.quiver(x, y, z, rot[0, i], rot[1, i], rot[2, i], color=colors[i])
    return ax


# example output of tag detection
# {'tag_family': b'tag36h11',
#  'tag_id': 4,
#  'hamming': 0,
#  'decision_margin': 89.33907318115234,
#  'homography': array([[-3.88145477e-01, -2.26832111e-01, 3.03084739e+02],
#                       [-3.94010603e+01, 7.38719243e+01, 2.24798384e+02],
#                       [-1.75052580e-01, -8.19634152e-04, 1.00000000e+00]]),
#  'center': array([303.08473899, 224.79838449]),
#  'corners': array([[258.25033569, 287.90826416],
#                    [367.01803589, 314.59835815],
#                    [366.83883667, 135.05673218],
#                    [258.27612305, 161.86071777]]),
#  'pose_R': array([[7.03999029e-01, 2.60361403e-04, 7.10200887e-01],
#                   [-2.34253507e-04, 9.99999964e-01, -1.34394584e-04],
#                   [-7.10200896e-01, -7.17533919e-05, 7.03999064e-01]]),
#  'pose_t': array([[1.07547828e-02],
#                   [-4.99777393e-04],
#                   [1.05013076e+00]]),
#  'pose_err': 1.197725413617014e-07}

# @dataclass
class TrainData:
    def __init__(self, CamPoses: List[Pose3],
                 Keypoints: List[np.ndarray],
                 Descriptors: List[np.ndarray],
                 PointsXYZ: List[np.ndarray],
                 Tags: List[List],
                 TagID2ImgID: dict,
                 TagOrder2Poses: dict,
                 ProjectMat: np.ndarray):
        self.CamPoses = CamPoses
        self.Keypoints = Keypoints
        self.Descriptors = Descriptors
        self.PointsXYZ = PointsXYZ
        self.Tags = Tags
        self.TagID2ImgID = TagID2ImgID
        self.TagOrder2Poses = TagOrder2Poses
        self.ProjectMat = ProjectMat

    @classmethod
    def load_data(cls, load_dir: str):
        cam_proj = np.load(file=os.path.join(load_dir, "ProjectMat.npy"))
        if os.path.exists(os.path.join(load_dir, "TagID2ImgID.msgpk")):
            with open(os.path.join(load_dir, "TagID2ImgID.msgpk"), "rb") as data_file:
                byte_data = data_file.read()
            tagid2imgid = msgpack.unpackb(byte_data,strict_map_key=False)
        else:
            tagid2imgid = np.load(os.path.join(load_dir, "TagID2ImgID.npy"), allow_pickle=True).item()

        with open(os.path.join(load_dir, "TagOrder2Poses.msgpk"), "rb") as data_file:
            byte_data = data_file.read()
        tagid2mat = msgpack.unpackb(byte_data,strict_map_key=False)
        tagid2pose = {}
        for key in tagid2mat:
            tagid2pose[key] = Pose3(tagid2mat[key])
        print("loaded project mat, tagid2imgid, gt markers")

        cam_mats = np.load(file=os.path.join(load_dir, "CamPoses.npy"))
        cam_poses = [Pose3(mat) for mat in cam_mats]
        dft_len = len(cam_poses)
        print("loaded camera poses")

        with open(os.path.join(load_dir, "Tags.msgpk"), "rb") as data_file:
            byte_data = data_file.read()
        tags = msgpack.unpackb(byte_data)
        for tag_arr in tags:
            for tmp_i in range(len(tag_arr)):
                tmp_tag = Detection()
                tmp_tag.tag_id = tag_arr[tmp_i]['tag_id']
                tmp_tag.tag_family = tag_arr[tmp_i]['tag_family']
                tmp_tag.hamming = tag_arr[tmp_i]['hamming']
                tmp_tag.decision_margin = tag_arr[tmp_i]['decision_margin']
                tmp_tag.homography = tag_arr[tmp_i]['homography']
                tmp_tag.center = tag_arr[tmp_i]['center']
                tmp_tag.corners = tag_arr[tmp_i]['corners']
                tmp_tag.pose_err = tag_arr[tmp_i]['pose_err']
                tmp_tag.pose_t = np.array(tag_arr[tmp_i]['pose_t'])
                tmp_tag.pose_R = np.array(tag_arr[tmp_i]['pose_R'])
                tag_arr[tmp_i] = tmp_tag
        print("loaded tags")

        kp_num_list = np.load(os.path.join(load_dir, "KpLenList.npy"))

        kpt_list = []
        xyz_list = []
        des_list = []
        cur_idx = 0
        kpt_arr = np.load(file=os.path.join(load_dir, "Keypoints.npy"))
        xyz_arr = np.load(file=os.path.join(load_dir, "PointsXYZ.npy"))
        des_arr = np.load(file=os.path.join(load_dir, "Descriptors.npy"))
        for kp_num in kp_num_list:
            if kp_num == 0 :
                kpt_list.append([])
                xyz_list.append([])
                des_list.append([])
            else:
                kpt_list.append(kpt_arr[cur_idx:cur_idx+kp_num])
                xyz_list.append(xyz_arr[cur_idx:cur_idx+kp_num])
                des_list.append(des_arr[cur_idx:cur_idx+kp_num])
            cur_idx += kp_num
        print("loaded all data")
        return cls(cam_poses, kpt_list, des_list, xyz_list, tags, tagid2imgid, tagid2pose, cam_proj)

    def save_data(self, save_dir: str):
        print("serializing small files")

        tagid2imgid = deepcopy(self.TagID2ImgID)
        tagid2mat = {}
        for tmp_k, tmp_v in self.TagOrder2Poses.items():
            tagid2mat[tmp_k] = tmp_v.mat.tolist()

        np.save(os.path.join(save_dir, "TagID2ImgID.npy"), tagid2imgid)

        # with open(os.path.join(save_dir, "TagID2ImgID.msgpk"), "wb") as outfile:
        #     packed = msgpack.packb(tagid2imgid, use_bin_type=True)
        #     outfile.write(packed)
        with open(os.path.join(save_dir, "TagOrder2Poses.msgpk"), "wb") as outfile:
            packed = msgpack.packb(tagid2mat, use_bin_type=True)
            outfile.write(packed)
        print("saved project mat, tagid2imgid, gt markers")

        dft_len = len(self.CamPoses)
        cam_mat_list = np.array([tmp_pose.mat for tmp_pose in self.CamPoses])
        np.save(file=os.path.join(save_dir, "CamPoses.npy"), arr=cam_mat_list)
        print("saved camera poses")

        tag_list = []
        for i in tqdm(range(dft_len)):
            tagsInFrame = self.Tags[i]
            if len(tagsInFrame) > 0:
                tag_dicts = []
                for tmp_tag_obj in tagsInFrame:
                    tmp_tag = deepcopy(tmp_tag_obj.__dict__)
                    tmp_tag['homography'] = tmp_tag['homography'].tolist()
                    tmp_tag['center'] = tmp_tag['center'].tolist()
                    tmp_tag['corners'] = tmp_tag['corners'].tolist()
                    tmp_tag['pose_R'] = tmp_tag['pose_R'].tolist()
                    tmp_tag['pose_t'] = tmp_tag['pose_t'].tolist()
                    tag_dicts.append(tmp_tag)
                tag_list.append(tag_dicts)
            else:
                tag_list.append([])
        with open(os.path.join(save_dir, "Tags.msgpk"), "wb") as outfile:
            packed = msgpack.packb(tag_list, use_bin_type=True)
            outfile.write(packed)
        del tag_list
        gc.collect()
        print("saved tags")

        assert len(self.Keypoints) == dft_len
        kpt_list = []
        xyz_list = []
        des_list = []
        kp_len_list = []
        print("saving pixels of key points")
        for i in tqdm(range(dft_len)):
            # assert len(kpt_list[i]) == len(des_list[i]) == len(xyz_list[i])
            kp_len_list.append(len(self.Keypoints[i]))
            if kp_len_list[-1] > 0:
                kpt_list.append(self.Keypoints[i])
                xyz_list.append(self.PointsXYZ[i])
                des_list.append(self.Descriptors[i])
        np.save(file=os.path.join(save_dir, "KpLenList.npy"), arr=kp_len_list)
        np.save(file=os.path.join(save_dir, "Keypoints.npy"), arr=np.vstack(kpt_list))
        np.save(file=os.path.join(save_dir, "PointsXYZ.npy"), arr=np.vstack(xyz_list))
        np.save(file=os.path.join(save_dir, "Descriptors.npy"), arr=np.vstack(des_list))

        cam_mat = self.ProjectMat
        np.save(file=os.path.join(save_dir, "ProjectMat.npy"), arr=cam_mat)
        print("saved all data")

# @dataclass
class TestData:
    def __init__(self,
        CamPoses: List[Pose3],
        Keypoints: List[np.ndarray],
        Descriptors: List[np.ndarray],
        Tags: List[List],
        TagID2ImgID: dict,
        ProjectMat: np.ndarray):
            self.CamPoses = CamPoses
            self.Keypoints = Keypoints
            self.Descriptors = Descriptors
            self.Tags = Tags
            self.TagID2ImgID = TagID2ImgID
            self.ProjectMat = ProjectMat

    @classmethod
    def load_data(cls, load_dir: str):
        cam_proj = np.load(file=os.path.join(load_dir, "ProjectMat.npy"))
        if os.path.exists(os.path.join(load_dir, "TagID2ImgID.msgpk")):
            with open(os.path.join(load_dir, "TagID2ImgID.msgpk"), "rb") as data_file:
                byte_data = data_file.read()
            tagid2imgid = msgpack.unpackb(byte_data,strict_map_key=False)
        else:
            tagid2imgid = np.load(os.path.join(load_dir, "TagID2ImgID.npy"), allow_pickle=True).item()
        print("loaded project mat and tagid2imgid")

        cam_mats = np.load(file=os.path.join(load_dir, "CamPoses.npy"))
        cam_poses = [Pose3(mat) for mat in cam_mats]
        print("loaded camera poses")

        with open(os.path.join(load_dir, "Tags.msgpk"), "rb") as data_file:
            byte_data = data_file.read()
        tags = msgpack.unpackb(byte_data)
        for tag_arr in tags:
            for tmp_i in range(len(tag_arr)):
                tmp_tag = Detection()
                tmp_tag.tag_id = tag_arr[tmp_i]['tag_id']
                tmp_tag.tag_family = tag_arr[tmp_i]['tag_family']
                tmp_tag.hamming = tag_arr[tmp_i]['hamming']
                tmp_tag.decision_margin = tag_arr[tmp_i]['decision_margin']
                tmp_tag.homography = tag_arr[tmp_i]['homography']
                tmp_tag.center = tag_arr[tmp_i]['center']
                tmp_tag.corners = tag_arr[tmp_i]['corners']
                tmp_tag.pose_err = tag_arr[tmp_i]['pose_err']
                tmp_tag.pose_t = np.array(tag_arr[tmp_i]['pose_t'])
                tmp_tag.pose_R = np.array(tag_arr[tmp_i]['pose_R'])
                tag_arr[tmp_i] = tmp_tag
        print("loaded tags")

        kp_num_list = np.load(os.path.join(load_dir, "KpLenList.npy"))

        kpt_list = []
        des_list = []
        cur_idx = 0
        kpt_arr = np.load(file=os.path.join(load_dir, "Keypoints.npy"))
        des_arr = np.load(file=os.path.join(load_dir, "Descriptors.npy"))
        for kp_num in kp_num_list:
            if kp_num == 0 :
                kpt_list.append([])
                des_list.append([])
            else:
                kpt_list.append(kpt_arr[cur_idx:cur_idx+kp_num])
                des_list.append(des_arr[cur_idx:cur_idx+kp_num])
            cur_idx += kp_num
        print("loaded all data")
        return cls(cam_poses, kpt_list, des_list, tags, tagid2imgid, cam_proj)

    def save_data(self, save_dir: str):
        print("serializing small files")
        tagid2imgid = deepcopy(self.TagID2ImgID)

        np.save(os.path.join(save_dir, "TagID2ImgID.npy"), tagid2imgid)

        # with open(os.path.join(save_dir, "TagID2ImgID.msgpk"), "wb") as outfile:
        #     packed = msgpack.packb(tagid2imgid, use_bin_type=True)
        #     outfile.write(packed)
        print("saved project mat and tagid2imgid")

        dft_len = len(self.CamPoses)
        cam_mat_list = np.array([tmp_pose.mat for tmp_pose in self.CamPoses])
        np.save(file=os.path.join(save_dir, "CamPoses.npy"), arr=cam_mat_list)
        print("saved camera poses")

        tag_list = []
        for i in tqdm(range(dft_len)):
            tagsInFrame = self.Tags[i]
            if len(tagsInFrame) > 0:
                tag_dicts = []
                for tmp_tag_obj in tagsInFrame:
                    tmp_tag = deepcopy(tmp_tag_obj.__dict__)
                    tmp_tag['homography'] = tmp_tag['homography'].tolist()
                    tmp_tag['center'] = tmp_tag['center'].tolist()
                    tmp_tag['corners'] = tmp_tag['corners'].tolist()
                    tmp_tag['pose_R'] = tmp_tag['pose_R'].tolist()
                    tmp_tag['pose_t'] = tmp_tag['pose_t'].tolist()
                    tag_dicts.append(tmp_tag)
                tag_list.append(tag_dicts)
            else:
                tag_list.append([])
        with open(os.path.join(save_dir, "Tags.msgpk"), "wb") as outfile:
            packed = msgpack.packb(tag_list, use_bin_type=True)
            outfile.write(packed)
        del tag_list
        gc.collect()
        print("saved tags")

        assert len(self.Keypoints) == dft_len
        kpt_list = []
        des_list = []
        kp_len_list = []
        print("saving pixels of key points")
        for i in tqdm(range(dft_len)):
            # assert len(kpt_list[i]) == len(des_list[i]) == len(xyz_list[i])
            kp_len_list.append(len(self.Keypoints[i]))
            if kp_len_list[-1] > 0:
                kpt_list.append(self.Keypoints[i])
                des_list.append(self.Descriptors[i])
        np.save(file=os.path.join(save_dir, "KpLenList.npy"), arr=kp_len_list)
        np.save(file=os.path.join(save_dir, "Keypoints.npy"), arr=np.vstack(kpt_list))
        np.save(file=os.path.join(save_dir, "Descriptors.npy"), arr=np.vstack(des_list))

        cam_mat = self.ProjectMat
        np.save(file=os.path.join(save_dir, "ProjectMat.npy"), arr=cam_mat)
        print("saved all data")


def plot_frames(id2pose: dict, zlim=[-5, 5]):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for id, pose in id2pose.items():
        ax = plot_frame(ax, pose, label=str(id))
    ax.set_zlim(*zlim)
    return fig, ax


def plot_markers(id2pose: dict):
    fig, ax = plt.subplots()
    for id, pose in id2pose.items():
        arrow = markers.MarkerStyle(marker='$\u2193$')  # Downwards arrow in Unicode: ↓
        x, y, theta = pose.mat[0, 3], pose.mat[1, 3], np.arctan2(pose.mat[1, 2], pose.mat[0, 2])
        arrow._transform = arrow.get_transform().rotate_deg(90 + theta * 180 / np.pi)
        ax.scatter((x), (y), marker=arrow)
        ax.text((x + .5), (y - .5), str(id))
    return fig, ax

def pickleListInParts(mylist, f_name, t_dir, lenPerPart=1000):
    c_init = 0
    c_end = lenPerPart
    c_idx = 0
    while c_end < len(mylist):
        with open(f"{t_dir}/{f_name}.part{c_idx}.pkl", "wb") as f:
            pickle.dump(mylist[c_init: c_end], f, pickle.HIGHEST_PROTOCOL)
        c_init = c_end
        c_end += lenPerPart
        c_idx += 1
    with open(f"{t_dir}/{f_name}.part{c_idx}.pkl", "wb") as f:
        pickle.dump(mylist[c_init:], f, pickle.HIGHEST_PROTOCOL)

def loadListInParts(f_name, t_dir):
    prefix = f"{f_name}.part"
    part_files = [f for f in os.listdir(t_dir)
                 if os.path.isfile(os.path.join(t_dir, f)) and len(f) > len(prefix) and f[:len(prefix)] == prefix]
    res = []
    for i in range(len(part_files)):
        part_name = f"{prefix}{i}.pkl"
        with open(os.path.join(t_dir, part_name), "rb") as f:
            res += pickle.load(f)
    return res

def trainDataReader(parent_folder: str,
                    params: ParamStruct,
                    frame_num=None,
                    marker_prefix="marker",
                    save_marker_plot=False,
                    dep_type="Planar",
                    tag_size=None,
                    at_detector=None
                    ):
    # params
    if tag_size is None:
        tag_size = params.tag_size
    tag_family = params.tag_family
    depth_max = params.depth_max
    img_w = params.img_w
    img_h = params.img_h
    fov = params.fov
    aprilTagInBody = Pose3(params.tf_aprilTagInBody)
    bodyInW = Pose3(params.tf_bodyInW)
    camInB = Pose3(params.tf_camInB)
    dep_rgb_scale = params.dep_w / params.img_w

    # extract data from image
    c_path = os.path.join(parent_folder, "train", "airsim_rec.txt")
    m_path = os.path.join(parent_folder, "airsim_marker.txt")
    c_data = np.loadtxt(c_path, delimiter='\t', dtype=object)
    m_data = []
    if os.path.exists(m_path):
        m_data = np.loadtxt(m_path, delimiter='\t', dtype=object)

    # w, h, fov = 1200, 900, 90
    focal_len = img_w / 2 / np.tan(np.pi * fov / 360)
    cameraMatrix = np.array([[focal_len, 0, img_w / 2],
                             [0, focal_len, img_h / 2],
                             [0, 0, 1]])

    # scan data
    gt_markers = {}
    if marker_prefix is not None:
        for i in tqdm(range(1, len(m_data))):
            row = m_data[i]
            # MarkerName POS_X	POS_Y	POS_Z	Q_W Q_X	Q_Y	Q_Z
            m_id = int(row[0][len(marker_prefix):])
            if 'nan' not in row[1:]:
                T = [float(row[i]) for i in range(1, 8)]
                pose = bodyInW * Pose3.by_vec(*T) * aprilTagInBody
                gt_markers[m_id] = pose
    if save_marker_plot:
        fig, ax = plot_markers(gt_markers)
        fig.savefig(os.path.join(parent_folder, "train_marker.png"), dpi=300)

    #### test WITH THE SAMPLE IMAGE ####
    camera_params = (cameraMatrix[0, 0], cameraMatrix[1, 1], cameraMatrix[0, 2], cameraMatrix[1, 2])
    # at_detector = Detector(families=tag_family,
    #                        nthreads=1,
    #                        quad_decimate=1.0,
    #                        quad_sigma=0.0,
    #                        refine_edges=1,
    #                        decode_sharpening=0.25,
    #                        debug=0)
    # create SIFT
    # sift = cv.SIFT.create()
    if params.dft_sift:
        sift = cv.SIFT.create()
    else:
        sift = cv.SIFT.create(nfeatures=params.preprocess_nfeature,
                              contrastThreshold=params.preprocess_contrastTh,
                              edgeThreshold=params.preprocess_edgeTh)

    marker2imgID = {}
    train_poses = []
    train_kps = []
    train_3dpt = []
    train_des = []
    train_tags = []

    if frame_num is None:
        f_n = len(c_data) - 1
    else:
        f_n = frame_num

    for img_id in tqdm(range(f_n)): #f_n)): #f_n)):
        row = c_data[img_id + 1]
        # VehicleName	TimeStamp	POS_X	POS_Y	POS_Z	Q_W	Q_X	Q_Y	Q_Z	ImageFile FilePath
        T = [float(row[i]) for i in range(2, 9)]
        body_pose = bodyInW * Pose3.by_vec(*T)
        cam_pose = body_pose * camInB

        img_folder = row[-1]
        dep_path = os.path.join(img_folder, f"DEP{img_id}.pfm")
        rgb_path = os.path.join(img_folder, f"RGB{img_id}.png")

        # images
        gray = cv.imread(rgb_path, cv.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Read again {rgb_path}.")
            gray = cv.imread(rgb_path, cv.IMREAD_GRAYSCALE)
        tags = []
        # tag detections
        try:
            tags = at_detector.detect(gray, True, camera_params, tag_size)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        for tag in tags:
            if tag.tag_id not in marker2imgID:
                marker2imgID[tag.tag_id] = [img_id]
            else:
                marker2imgID[tag.tag_id].append(img_id)
        # depth image
        dep_img = read_pfm(dep_path)

        # SIFT features
        try:
            kp, des = sift.detectAndCompute(gray, None)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            kp = []
            des = []

        raw_px_y_z, outdes, pt_w_xyz = [], [], []
        # get 3d positions and filter out very distant points
        if (len(kp) > 0) & (des is not None):
            px_z_y = np.array([[round(kp[i].pt[1]), round(kp[i].pt[0])] for i in range(len(kp))])
            px_d = depth_interpolation(dep_img, px_z_y, dep_rgb_scale)  # dep_img[tuple(px_z_y.T)]
            ptInFov = (px_d < depth_max)
            outkp = [kp[i] for i in range(len(kp)) if ptInFov[i]]
            outdes = des[ptInFov]
            if len(outkp) > 0:
                px_z_y = px_z_y[ptInFov]
                px_d = px_d[ptInFov]
                body_xyz = px2coor(img_w, img_h, px_z_y[:, 1], px_z_y[:, 0], px_d, focal_len, dep_type)
                body_xyz = np.hstack((body_xyz, np.ones((body_xyz.shape[0], 1))))
                pt_w_xyz = (body_pose.mat @ body_xyz.T).T[:, :3]
                raw_px_y_z = np.array([[pt.pt[0], pt.pt[1]] for pt in outkp])
        train_des.append(outdes)
        train_kps.append(raw_px_y_z)
        train_poses.append(cam_pose)
        train_tags.append(tags)
        train_3dpt.append(pt_w_xyz)
    # time.sleep(10.0)
    return TrainData(train_poses, train_kps, train_des, train_3dpt, train_tags, marker2imgID, gt_markers, cameraMatrix)
    # trainData = TrainData(train_poses, train_kps, train_des, train_3dpt, train_tags, marker2imgID, gt_markers,
    #                       cameraMatrix)
    # return trainData


def visualDictionary(descriptors, numVisualWords):
    return kMeansDictionary(descriptors, numVisualWords)


def getVLADDescriptors(local_descriptors, visualDictionary, parallel_jobs=4):
    return Parallel(n_jobs=parallel_jobs)(delayed(VLAD)(des, visualDictionary) for des in local_descriptors)


def testDataReader(case_folder: str,
                   params: ParamStruct,
                   frame_num=None,
                   tag_size=None,
                   at_detector=None
                   ):
    # params
    if tag_size is None:
        tag_size = params.tag_size
    tag_family = params.tag_family
    img_w = params.img_w
    img_h = params.img_h
    fov = params.fov
    bodyInW = Pose3(params.tf_bodyInW)
    camInB = Pose3(params.tf_camInB)

    # extract data from image
    c_path = os.path.join(case_folder, "airsim_rec.txt")
    c_data = np.loadtxt(c_path, delimiter='\t', dtype=object)

    # depth images
    # w, h, fov = 1200, 900, 90
    focal_len = img_w / 2 / np.tan(np.pi * fov / 360)
    cameraMatrix = np.array([[focal_len, 0, img_w / 2],
                             [0, focal_len, img_h / 2],
                             [0, 0, 1]])
    # create SIFT
    # sift = cv.SIFT.create()
    if params.dft_sift:
        sift = cv.SIFT.create()
    else:
        sift = cv.SIFT.create(nfeatures=params.preprocess_nfeature,
                              contrastThreshold=params.preprocess_contrastTh,
                              edgeThreshold=params.preprocess_edgeTh)

    # tag detection settings
    # markers
    # at_detector = Detector(families=tag_family,
    #                        nthreads=1,
    #                        quad_decimate=1.0,
    #                        quad_sigma=0.0,
    #                        refine_edges=1,
    #                        decode_sharpening=0.25,
    #                        debug=0)

    #### test WITH THE SAMPLE IMAGE ####
    camera_params = (cameraMatrix[0, 0], cameraMatrix[1, 1], cameraMatrix[0, 2], cameraMatrix[1, 2])

    marker2imgID = {}
    test_poses = []
    test_kps = []
    test_des = []
    test_tags = []

    if frame_num is None:
        f_n = len(c_data) - 1
    else:
        f_n = frame_num

    for img_id in tqdm(range(f_n)): # f_n)):
        row = c_data[img_id + 1]
        # VehicleName	TimeStamp	POS_X	POS_Y	POS_Z	Q_W	Q_X	Q_Y	Q_Z	ImageFile
        T = [float(row[i]) for i in range(2, 9)]
        body_pose = bodyInW * Pose3.by_vec(*T)
        cam_pose = body_pose * camInB
        rgb_path = os.path.join(row[-1], f"RGB{img_id}.png")

        # images
        gray = cv.imread(rgb_path, cv.IMREAD_GRAYSCALE)
        # tag detections
        try:
            tags = at_detector.detect(gray, True, camera_params, tag_size)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            tags = []
        for tag in tags:
            if tag.tag_id not in marker2imgID:
                marker2imgID[tag.tag_id] = [img_id]
            else:
                marker2imgID[tag.tag_id].append(img_id)
        # SIFT features
        try:
            kp, des = sift.detectAndCompute(gray, None)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            kp = []
            des = []

        # raw_px_y_z, outdes, pt_w_xyz = [], [], []
        # # get 3d positions and filter out very distant points
        # if (len(kp) > 0) & (des is not None):
        #     px_z_y = np.array([[round(kp[i].pt[1]), round(kp[i].pt[0])] for i in range(len(kp))])
        #     px_d = depth_interpolation(dep_img, px_z_y, dep_rgb_scale)  # dep_img[tuple(px_z_y.T)]
        #     ptInFov = (px_d < depth_max)
        #     outkp = [kp[i] for i in range(len(kp)) if ptInFov[i]]
        #     outdes = des[ptInFov]
        #     if len(outkp) > 0:
        #         px_z_y = px_z_y[ptInFov]
        #         px_d = px_d[ptInFov]
        #         body_xyz = px2coor(img_w, img_h, px_z_y[:, 1], px_z_y[:, 0], px_d, focal_len, dep_type)
        #         body_xyz = np.hstack((body_xyz, np.ones((body_xyz.shape[0], 1))))
        #         pt_w_xyz = (body_pose.mat @ body_xyz.T).T[:, :3]
        #         raw_px_y_z = np.array([[pt.pt[0], pt.pt[1]] for pt in outkp])

        if des is None:
            des = []

        raw_px_y_z = np.array([[kp[i].pt[0], kp[i].pt[1]] for i in range(len(kp))])
        test_des.append(des)
        test_kps.append(raw_px_y_z)
        test_poses.append(cam_pose)
        test_tags.append(tags)
    # time.sleep(10.0)
    return TestData(test_poses, test_kps, test_des, test_tags, marker2imgID, cameraMatrix)


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
                  flann_k=2, ratio_test=.8, rank_tags="vlad", tag_loc_percent=0.5, rd_seed=1, matchedImgID=[], bf_match=True):
    if des is None or len(des) == 0:
        return None
    else:
        if isinstance(des, list):
            des = np.array([des])
        tag_loc_nn = max(0, min(int(numNNs * tag_loc_percent), numNNs))

        tag_matched_train_id = set()
        all_matched_train_id = set()
        # if test_img_id == 40:
        #     print("test_img_id", 40)
        if tag_loc_nn>0:
            for tag in tags:
                if tag.tag_id in train_data.TagID2ImgID:
                    tag_matched_train_id.update(train_data.TagID2ImgID[tag.tag_id])
                # get train camera poses that are close to the projected test camera pose
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
                # pick train images that have more features
                if rank_tags == "feature_num":
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
        # if len(train_data.Tags[train_id]) > 0:
        #     kp_xy = np.array([[kkp[0], kkp[1]] for kkp in train_kp])
        #     outtag = np.ones(len(kp_xy), dtype=bool)
        #     for tag_det in train_data.Tags[train_id]:
        #         assert isinstance(tag_det, Detection)
        #         if tag_det.tag_id in test_ids:
        #             continue
        #         corners = tag_det.corners
        #         p = path.Path(corners)
        #         inside = p.contains_points(kp_xy)
        #         outtag = outtag & (~inside)
        #     train_kp = [kkp for kp_i, kkp in enumerate(train_kp) if outtag[kp_i]]
        #     if sum(outtag) == 0:
        #         train_des = []
        #     else:
        #         train_des = np.vstack([kkp for kp_i, kkp in enumerate(train_des) if outtag[kp_i]])
        #     train_xyz = [kkp for kp_i, kkp in enumerate(train_xyz) if outtag[kp_i]]
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
        # increase 200 to 150000/100000/50000 or default
        # plain p3p
        # solved, rvec, tvec, inliers = cv.solvePnPRansac(pts3d, pts2d, train_data.ProjectMat, np.zeros(4), iterationsCount=200, reprojectionError=4.0)
        # solved, rvec, tvec, inliers = cv.solvePnPRansac(pts3d, pts2d, train_data.ProjectMat, np.zeros(4),
        #                                                 iterationsCount=200, reprojectionError=4.0,
        #                                                 flags=cv.SOLVEPNP_P3P)
        # tvec = test_cam_pose.inverse().translation
        # rvec = cv.Rodrigues(test_cam_pose.inverse().rotation)[0]
        # solved, rvec, tvec, inliers = cv.solvePnPRansac(pts3d, pts2d, train_data.ProjectMat, np.zeros(4),
        #                                                 iterationsCount=200, reprojectionError=4.0,
        #                                                 useExtrinsicGuess=True, rvec=rvec, tvec=tvec,
        #                                                 flags=cv.SOLVEPNP_ITERATIVE)
        if solved:
            retval, rvec, tvec = cv.solvePnP(pts3d[inliers], pts2d[inliers], train_data.ProjectMat, np.zeros(4),
                                             useExtrinsicGuess=True, rvec=rvec, tvec=tvec, flags=cv.SOLVEPNP_ITERATIVE)
            # solved, rvec, tvec, inliers = cv.solvePnPRansac(pts3d, pts2d, train_data.ProjectMat, np.zeros(4),
            #                                                 iterationsCount=200, reprojectionError=4.0,
            #                                                 useExtrinsicGuess=True, rvec=rvec, tvec=tvec,
            #                                                 flags=cv.SOLVEPNP_ITERATIVE)
            # if solved:
            # rvec2, tvec2 = cv.solvePnPRefineLM(pts3d[inliers.flatten()], pts2d[inliers.flatten()], train_data.ProjectMat, np.zeros(4), rvec, tvec)
            # print(f"refine diff {tvec2 - tvec}")
            # rvec, tvec=cv.solvePnPRefineLM(pts3d, pts2d, train_data.ProjectMat, np.zeros(4), rvec=rvec, tvec=tvec)
            mat = np.eye(4)
            mat[:3, :3] = cv.Rodrigues(rvec)[0]
            mat[:3, 3] = tvec.flatten()
            pose = Pose3(mat).inverse()
            # t_err = np.linalg.norm(pose.translation - test_cam_pose.translation)
            # temp = np.clip((np.trace(pose.rotation.T @ test_cam_pose.rotation) - 1) / 2, -1, 1)
            # R_err = abs(np.arccos(temp)) * 180 / np.pi
            # print(f"test img {test_img_id} rotation err {R_err} translation err {t_err}")
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

    # res = axes[1].scatter(camPoints[:, 0], camPoints[:, 1], marker='.', s=100,
    #                  c=entropy_list, cmap=cmap)
    return fig, axes


def get_id_vlad_des(des_list, prefix, pkl_folder, numVisualWords):
    train_des_id = [i for i, des in enumerate(des_list) if des is not None and len(des) > 0]
    valid_des = [des_list[i] for i in train_des_id]
    visDict_pkl = os.path.join(pkl_folder, "visual_dict.pkl")
    if not os.path.exists(visDict_pkl):
        visual_dict = visualDictionary(np.vstack(valid_des), numVisualWords)
        with open(visDict_pkl, "wb") as f:
            pickle.dump(visual_dict, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(visDict_pkl, "rb") as f:
            visual_dict = pickle.load(f)

    # compute vlad for tag images
    train_vlad_pkl = os.path.join(pkl_folder, f"{prefix}_vlad.pkl")
    if not os.path.exists(train_vlad_pkl):
        print("computing vlad descriptors...")
        train_vlad_des = getVLADDescriptors(valid_des, visual_dict)
        imgID2vladDes = {train_des_id[i]: train_vlad_des[i] for i in range(len(train_des_id))}
        with open(train_vlad_pkl, "wb") as f:
            pickle.dump(imgID2vladDes, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(train_vlad_pkl, "rb") as f:
            imgID2vladDes = pickle.load(f)
        train_vlad_des = [imgID2vladDes[des_id] if des_id in imgID2vladDes
                          else getVLADDescriptors([des_list[des_id]],visual_dict)
                          for des_id in train_des_id]
        train_vlad_des = np.vstack(train_vlad_des)
    return train_des_id, train_vlad_des


def get_test_data(pkl_folder, case_folder, test_folder, params, test_frame_num, tag_size=None, at_detector=None):
    raw_data_dir = os.path.join(case_folder, test_folder)
    test_pkl = os.path.join(pkl_folder, f'{test_folder}_data.pkl')
    save_dir = os.path.join(pkl_folder, f'{test_folder}_data')
    if os.path.exists(test_pkl):
        with open(test_pkl, "rb") as f:
            test_data = pickle.load(f)
    elif os.path.exists(save_dir) and os.path.exists(f"{save_dir}/ProjectMat.pkl"):
        cam_pose_pkl = f"{save_dir}/CamPoses.pkl"
        if os.path.exists(cam_pose_pkl):
            with open(cam_pose_pkl, "rb") as f:
                CamPoses = pickle.load(f)
        else:
            cam_pose_npy = f"{save_dir}/CamPoses.npy"
            cam_arr = np.load(cam_pose_npy)
            CamPoses = [Pose3(pose_mat) for pose_mat in cam_arr]
        with open(f"{save_dir}/Keypoints.pkl", "rb") as f:
            Keypoints = pickle.load(f)
        # with open(f"{save_dir}/Descriptors.pkl", "rb") as f:
        #     Descriptors = pickle.load(f)
        des_prefix = "Descriptors"
        des_pkl_file = os.path.join(save_dir, f"{des_prefix}.pkl")
        des_npy_file = os.path.join(save_dir, f"{des_prefix}.npy")
        if os.path.exists(des_pkl_file):
            with open(des_pkl_file, "rb") as f:
                Descriptors = pickle.load(f)
        elif os.path.exists(des_npy_file):
            Descriptors = np.load(des_npy_file)
        else:
            # descriptor stored in parts
            Descriptors = loadListInParts(des_prefix, save_dir)
        with open(f"{save_dir}/Tags.pkl", "rb") as f:
            Tags = pickle.load(f)
        with open(f"{save_dir}/TagID2ImgID.pkl", "rb") as f:
            TagID2ImgID = pickle.load(f)
        with open(f"{save_dir}/ProjectMat.pkl", "rb") as f:
            ProjectMat = pickle.load(f)
        test_data = TestData(CamPoses, Keypoints, Descriptors, Tags, TagID2ImgID, ProjectMat)
    elif os.path.exists(save_dir) and os.path.exists(f"{save_dir}/ProjectMat.npy"):
        test_data = TestData.load_data(save_dir)
    else:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        test_data = testDataReader(raw_data_dir, params, test_frame_num,tag_size=tag_size,at_detector=at_detector)
        print("get data")
        print("start saving")
        test_data.save_data(save_dir)
    return test_data


def get_train_data(pkl_folder, case_folder, params, train_frame_num, marker_name, save_marker_plot, tag_size=None,at_detector=None):
    train_pkl = os.path.join(pkl_folder, 'train_data.pkl')
    train_dir = os.path.join(pkl_folder, 'train_data')
    if os.path.exists(train_pkl):
        with open(train_pkl, "rb") as f:
            train_data = pickle.load(f)
    elif os.path.exists(train_dir) and os.path.exists(f"{train_dir}/ProjectMat.pkl"):
        cam_pose_pkl = f"{train_dir}/CamPoses.pkl"
        if os.path.exists(cam_pose_pkl):
            with open(cam_pose_pkl, "rb") as f:
                CamPoses = pickle.load(f)
        else:
            cam_pose_npy = f"{train_dir}/CamPoses.npy"
            cam_arr = np.load(cam_pose_npy)
            CamPoses = [Pose3(pose_mat) for pose_mat in cam_arr]
        with open(f"{train_dir}/Keypoints.pkl", "rb") as f:
            Keypoints = pickle.load(f)

        des_prefix = "Descriptors"
        des_pkl_file = os.path.join(train_dir, f"{des_prefix}.pkl")
        des_npy_file = os.path.join(train_dir, f"{des_prefix}.npy")
        if os.path.exists(des_pkl_file):
            with open(des_pkl_file, "rb") as f:
                Descriptors = pickle.load(f)
        elif os.path.exists(des_npy_file):
            Descriptors = np.load(des_npy_file)
        else:
            # descriptor stored in parts
            Descriptors = loadListInParts(des_prefix, train_dir)

        with open(f"{train_dir}/PointsXYZ.pkl", "rb") as f:
            PointsXYZ = pickle.load(f)
        with open(f"{train_dir}/Tags.pkl", "rb") as f:
            Tags = pickle.load(f)
        with open(f"{train_dir}/TagID2ImgID.pkl", "rb") as f:
            TagID2ImgID = pickle.load(f)
        with open(f"{train_dir}/TagOrder2Poses.pkl", "rb") as f:
            TagOrder2Poses = pickle.load(f)
        with open(f"{train_dir}/ProjectMat.pkl", "rb") as f:
            ProjectMat = pickle.load(f)
        train_data = TrainData(CamPoses, Keypoints, Descriptors, PointsXYZ,
                               Tags, TagID2ImgID, TagOrder2Poses, ProjectMat)
    elif os.path.exists(train_dir) and os.path.exists(f"{train_dir}/ProjectMat.npy"):
        train_data = TrainData.load_data(train_dir)
    else:
        train_data = trainDataReader(case_folder, params, train_frame_num, marker_prefix=marker_name,save_marker_plot=save_marker_plot, tag_size=tag_size,at_detector=at_detector)
        print("get data")
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        print("start saving")
        train_data.save_data(train_dir)
    return train_data


if __name__ == "__main__":
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
            print(f"processing {c_name} with {m_num} markers...")
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
            # preparing training data (the map data)
            train_frame_num = None
            test_frame_num = None
            train_data = get_train_data(pkl_folder, case_folder, params, train_frame_num, marker_name, save_marker_plot,at_detector=at_detector)
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
                test_data = get_test_data(pkl_folder, case_folder, test_folder, params, test_frame_num,at_detector=at_detector)

                if use_nomarker_vlad:
                    test_des_id, test_vlad_des = get_id_vlad_des(test_data.Descriptors, test_folder,
                                                                 os.path.join(eva_folder, no_marker_case, "pkl"), numVisualWords)
                else:
                    test_des_id, test_vlad_des = get_id_vlad_des(test_data.Descriptors, test_folder, pkl_folder, numVisualWords)

                # del test_data
                gc.collect()
            # del train_data
            gc.collect()
