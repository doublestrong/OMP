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

    use_seg_img = False

    if reset_seg:
        # segmentations
        ## show segmentation IDs
        objects = client.simListSceneObjects()
        for ob in objects:
            print(ob)
        set_seg = True
        # IDs correspond to colors in this list:  # https://microsoft.github.io/AirSim/seg_rgbs.txt
        bg_ID = 0 # color [0, 0, 0]
        obj_ID = yaml_setting.wall_seg_id # color [153, 108, 6]
        object_name_list = yaml_setting.wall_labels
        if set_seg:
            print("Reset all object id")
            found = client.simSetSegmentationObjectID("[\w]*", 0, True)
            print("all object: %r" % (found))
            time.sleep(1.0)
            for idx, obj_name in enumerate(object_name_list):
                obj_name_reg = r"[\w]*" + obj_name + r"[\w]*"
                found = client.simSetSegmentationObjectID(obj_name_reg, obj_ID, True)
                use_seg_img = use_seg_img or found
                print("%s: %r" % (obj_name, found))
    print("Sleep for seconds")
    time.sleep(10.0)

    # rotate the camera to scan on the XY of these cubes
    free_pt_file = os.path.join(tmp_dir, "free_dot.txt")
    if os.path.exists(free_pt_file):
        obs_pts = np.loadtxt(free_pt_file)
    else:
        raise ValueError("No free dots.")
    work_dir = os.path.join(tmp_dir, "free_dots_scan_images")
    img_dir = os.path.join(work_dir, "images")
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        gt_str = "\t".join(["VehicleName", "TimeStamp",	"POS_X", "POS_Y","POS_Z", "Q_W", "Q_X", "Q_Y", "Q_Z", "ImageFile"])
        with open(os.path.join(work_dir, "airsim_rec.txt"), "w+") as f:
            f.writelines(gt_str)
        gt_str = ""

        angle_list = np.linspace(0, 2 * np.pi, yaml_setting.fine_angular_resolution, endpoint=False)
        cur_img_id = 0
        for pt_id, pt in enumerate(obs_pts):
            cur_angle_idx = 0
            while (cur_angle_idx < len(angle_list)):
                x = angle_list[cur_angle_idx]
                client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pt[0], pt[1], 0), airsim.to_quaternion(0, 0, -x)), True)
                time.sleep(.1)

                print(f"recording at dot {pt_id} and angle {cur_angle_idx}")

                if use_seg_img:
                    responses = client.simGetImages([
                        airsim.ImageRequest(cam_name, airsim.ImageType.Scene),
                        airsim.ImageRequest(cam_name, airsim.ImageType.DepthPlanar, pixels_as_float=True),
                        airsim.ImageRequest(cam_name, airsim.ImageType.Segmentation),
                        airsim.ImageRequest(cam_name, airsim.ImageType.SurfaceNormals)])
                    img_types = ["RGB", "DEP", "SEG", "SUR"]
                else:
                    responses = client.simGetImages([
                        airsim.ImageRequest(cam_name, airsim.ImageType.Scene),
                        airsim.ImageRequest(cam_name, airsim.ImageType.DepthPlanar, pixels_as_float=True),
                        airsim.ImageRequest(cam_name, airsim.ImageType.SurfaceNormals)])
                    img_types = ["RGB", "DEP", "SUR"]

                if None in responses:
                    print("None appears so re-do.")
                    continue
                img_names = []
                dep_response = None

                for img_i, img_type in enumerate(img_types):
                    if img_type == "DEP":
                        img_names.append("DEP" + str(cur_img_id) + ".pfm")
                        airsim.write_pfm(os.path.normpath(os.path.join(img_dir, img_names[-1])),
                                         airsim.get_pfm_array(responses[img_i]))
                        dep_response = responses[img_i]
                    else:
                        img_names.append(img_type + str(cur_img_id) + '.png')
                        airsim.write_file(os.path.normpath(os.path.join(img_dir, img_names[-1])),
                                          responses[img_i].image_data_uint8)
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
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)