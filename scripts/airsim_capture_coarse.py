# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
import airsim
import pprint
import os
import time
import numpy as np
import yaml
from utils import imp_setting

if __name__ == "__main__":
    yaml_setting = imp_setting("omp_settings.yaml")
    # prep
    client = airsim.VehicleClient()
    client.confirmConnection()
    pp = pprint.PrettyPrinter(indent=4)
    v_name = "myCam"
    cam_name = yaml_setting.cam_name
    reset_seg = True

    tmp_dir = yaml_setting.data_dir
    out_yaml = os.path.join(tmp_dir, "imp_setting.yaml")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    if not os.path.exists(out_yaml):
        with open(out_yaml, "w+") as stream:
            try:
                from yaml import CLoader as Loader, CDumper as Dumper
            except ImportError:
                from yaml import Loader, Dumper
            yaml.dump(yaml_setting.setting_dict, stream=stream, Dumper=Dumper)

    info_path = os.path.join(tmp_dir, "camera_info.txt")
    if not os.path.exists(info_path):
        # camera info
        camera_info = client.simGetCameraInfo(str(cam_name))
        camera_distort = client.simGetDistortionParams(str(cam_name))
        p1 = pp.pformat(camera_info)
        p2 = pp.pformat(camera_distort)
        print("CameraInfo %s: %s" % (cam_name, p1))
        print("CameraDistort %s: %s" % (cam_name, p2))
        with open(info_path, "w+") as f:
            f.write(pp.pformat(camera_info))
            f.write('\n')
            f.write(pp.pformat(camera_distort))

    setting_path = os.path.join(tmp_dir, "airsim_setting.txt")
    if not os.path.exists(setting_path):
        with open(setting_path, "w+") as f:
            f.write(client.getSettingsString())

    if reset_seg:
        # segmentations
        ## show segmentation IDs
        objects = client.simListSceneObjects()
        for ob in objects:
            print(ob)
        set_seg = True
        # IDs correspond to colors in this list:  # https://microsoft.github.io/AirSim/seg_rgbs.txt
        bg_ID = 0 # color [0, 0, 0]
        obj_ID = 1 # color [153, 108, 6]
        object_name_list = yaml_setting.wall_labels
        if set_seg:
            print("Reset all object id")
            found = client.simSetSegmentationObjectID("[\w]*", 0, True)
            print("all object: %r" % (found))
            time.sleep(1.0)
            for idx, obj_name in enumerate(object_name_list):
                obj_name_reg = r"[\w]*" + obj_name + r"[\w]*"
                found = client.simSetSegmentationObjectID(obj_name_reg, obj_ID, True)
                print("%s: %r" % (obj_name, found))
    print("Sleep for seconds")
    time.sleep(10.0)

    # get  x y locations of cubes where we do coarse scans for creating an occupancy grid
    # we export free dots from the occupancy grid
    cube_pt_file = os.path.join(tmp_dir, "cube_points.txt")
    cube_pt = []
    if os.path.exists(cube_pt_file):
        temp = np.loadtxt(cube_pt_file, dtype=str)
        print(f"Reading {cube_pt_file}: {temp[0]}")
        cube_pt = temp[1:, 1:].astype(float)
    else:
        # manually place cubes in the Unreal project
        base_str = str(yaml_setting.cube_label)
        obj_str = str(yaml_setting.cube_label)
        gt_str = "\t".join(["ObjID", "POS_X", "POS_Y"])
        next_cnt = 2
        while True:
            found = client.simGetObjectPose(obj_str)
            print("obj pose: %r" % (found))
            if str(found.position.x_val) != "nan":
                cube_pt.append([found.position.x_val, found.position.y_val])
                gt_str += "\n"
                gt_str += "\t".join([obj_str, str(found.position.x_val),
                                     str(found.position.y_val)])
                obj_str = base_str + str(next_cnt)
                next_cnt += 1
            else:
                break
        with open(cube_pt_file, "w+") as f:
            f.writelines(gt_str)
            gt_str = ""
        cube_pt = np.array(cube_pt)

    # rotate the camera to scan on the XY of these cubes
    work_dir = os.path.join(tmp_dir, "coarse_scan_images")
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
        img_dir = os.path.join(work_dir, "images")
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        gt_str = "\t".join(["VehicleName", "TimeStamp",	"POS_X", "POS_Y","POS_Z", "Q_W", "Q_X", "Q_Y", "Q_Z", "ImageFile"])
        with open(os.path.join(work_dir, "airsim_rec.txt"), "w+") as f:
            f.writelines(gt_str)
        gt_str = ""

        obs_pts = cube_pt.copy()

        angle_list = np.linspace(0, 2 * np.pi, yaml_setting.fine_angular_resolution, endpoint=False)
        cur_img_id = 0
        for pt_id, pt in enumerate(obs_pts):
            cur_angle_idx = 0
            while (cur_angle_idx < len(angle_list)):
                x = angle_list[cur_angle_idx]
                client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pt[0], pt[1], 0), airsim.to_quaternion(0, 0, -x)), True)
                time.sleep(0.1)

                print(f"recording at dot {pt_id} and angle {cur_angle_idx}")

                responses = client.simGetImages([
                    airsim.ImageRequest(cam_name, airsim.ImageType.Scene),
                    airsim.ImageRequest(cam_name, airsim.ImageType.DepthPlanar, pixels_as_float=True)])

                if None in responses:
                    print("None appears so re-do.")
                    continue
                img_names = []
                dep_response = responses[1]

                img_names.append("RGB" + str(cur_img_id) + '.png')
                airsim.write_file(os.path.normpath(os.path.join(img_dir, img_names[-1])), responses[0].image_data_uint8)

                img_names.append("DEP" + str(cur_img_id) + ".pfm")
                airsim.write_pfm(os.path.normpath(os.path.join(img_dir, img_names[-1])), airsim.get_pfm_array(dep_response))

                if dep_response is None:
                    print("None appears so re-do.")
                    continue

                gt_str += "\n"
                img_str = ';'.join(img_names)
                cur_line = "\t".join([v_name, str(dep_response.time_stamp), str(dep_response.camera_position.x_val),
                                      str(dep_response.camera_position.y_val), str(dep_response.camera_position.z_val),
                                      str(dep_response.camera_orientation.w_val), str(dep_response.camera_orientation.x_val),
                                      str(dep_response.camera_orientation.y_val), str(dep_response.camera_orientation.z_val), img_str])
                gt_str += cur_line
                with open(os.path.join(work_dir, "airsim_rec.txt"), "a") as f:
                    f.writelines(gt_str)
                gt_str = ""
                cur_angle_idx += 1
                cur_img_id += 1
                time.sleep(.1)