import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
from optimized_marker_placement import ResultCase, entropyScore, visualize, save_fig, informationScore
from utils import imp_setting, Pose3OnXY

if __name__ == "__main__":
    rd_cases_num = 5
    marker_num = 20
    case_folder = ""
    params = imp_setting(os.path.join(case_folder, "omp_settings.yaml"))

    xlim = [params.grid_x_offset, params.grid_x_offset + params.grid_x_len]
    ylim = [params.grid_y_offset, params.grid_y_offset + params.grid_y_len]

    random.seed(params.random_seed)
    save_dir = os.path.join(params.data_dir, "rd_placements")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    pickle_file = os.path.join(params.data_dir, 'candidate_marker_poses.pkl')
    free_dot_file = os.path.join(params.data_dir, 'free_dot.txt')
    scan_points = np.loadtxt(os.path.join(params.data_dir, 'scan_point_map.txt'))
    # note that free dots are in the NED frame (airsim conversion)
    free_dots = np.loadtxt(free_dot_file)
    free_dots[:, 1] *= -1
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            marker_poses = pickle.load(f)
        num_poses = len(marker_poses)
        idx_list = list(range(num_poses))
        for i in range(rd_cases_num):
            placed_id_file = os.path.join(save_dir, f"rd_placement{i}.txt")
            idx_sample = random.sample(idx_list, marker_num)
            placedMarkers = []
            with open(placed_id_file, "w+") as steeam:
                outstr = " ".join(['ID', 'x', 'y', 'theta'])
                for id in idx_sample:
                    pose = marker_poses[id]
                    placedMarkers.append(pose)
                    x, y, theta = Pose3OnXY(pose)
                    outstr += "\n" + " ".join([str(id), str(x), str(y), str(theta)])
                steeam.write(outstr)

            fig, ax = plt.subplots(figsize=(8, int(8*(ylim[1]-ylim[0]) / (xlim[1]-xlim[0]) )))
            fig, ax = visualize(fig, ax, free_dots, np.zeros(len(free_dots)), placedMarkers, [], [], cmap="gray")
            ax.scatter(scan_points[:,0], scan_points[:,1], color='red', s=.05)
            ax.set_title(
                f"Mean Localizability Score: 0.0")
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_aspect('equal', 'box')
            fig.savefig(f"{save_dir}/rd{i}.png", dpi=800, bbox_inches='tight')
            plt.close(fig)
    else:
        raise ValueError("pickle file of marker poses do not exist")