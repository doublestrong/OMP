import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P
import matplotlib.cm as cm
from tqdm import tqdm
from scipy.linalg import norm as sci_norm
from scipy.ndimage import binary_dilation as dilation
from utils import Pose3, read_pfm, px2coor, ray_bearing_width, imp_setting


class OccupMap():
    def __init__(self, xsize, ysize, x_offset, y_offset, grid_size, obj_thickness, beam_width, z_max = 30,
                 free_bearing_inflate=1.0, ocp_bearing_inflate=1.0):
        self.xsize = xsize+2 # Add extra cells for the borders
        self.ysize = ysize+2
        self.grid_size = grid_size # save this off for future use
        self.log_prob_map = np.zeros((self.ysize, self.xsize)) # set all to zero
        # self.state_map = np.zeros((self.xsize, self.ysize), dtype=int) # set all to zero

        self.alpha = obj_thickness #1.0 # The assumed thickness of obstacles
        self.beta = beam_width # 5.0*np.pi/180.0 # The assumed width of the laser beam
        self.z_max = z_max # The max reading from the laser

        # Pre-allocate the x and y positions of all grid positions into a 3D tensor
        # (pre-allocation = faster)
        self.grid_position_m = np.array([np.tile(np.arange(0, self.xsize*self.grid_size-0.0001, self.grid_size)[:,None].T + x_offset,
                                                 (self.ysize, 1)),
                                         np.tile(np.arange(0, self.ysize*self.grid_size-0.0001, self.grid_size)[:,None] + y_offset,
                                                 (1, self.xsize))])

        # grid state indicator: 0, unknown; 1, free; 2, occupied
        # self.free_state = 1
        # self.occupied_state = 2
        # Log-Probabilities to add or remove from the map
        self.l_occ = np.log(100)
        self.l_free = np.log(0.01)
        self.free_bearing_inflate = free_bearing_inflate
        self.ocp_bearing_inflate = ocp_bearing_inflate

    def update_map(self, pose, z):
        dx = self.grid_position_m.copy() # A tensor of coordinates of all cells
        dx[0, :, :] -= pose[0] # A matrix of all the x coordinates of the cell
        dx[1, :, :] -= pose[1] # A matrix of all the y coordinates of the cell
        theta_to_grid = np.arctan2(dx[1, :, :], dx[0, :, :]) - pose[2] # matrix of all bearings from robot to cell

        # Wrap to +pi / - pi
        theta_to_grid[theta_to_grid > np.pi] -= 2. * np.pi
        theta_to_grid[theta_to_grid < -np.pi] += 2. * np.pi

        dist_to_grid = sci_norm(dx, axis=0) # matrix of L2 distance to all cells from robot

        # For each laser beam
        for i, z_i in enumerate(z):
            r = z_i[0] # range measured
            b = z_i[1] # bearing measured
            if r < self.z_max:
                # Calculate which cells are measured free or occupied, so we know which cells to update
                # Doing it this way is like a billion times faster than looping through each cell (because vectorized numpy is the only way to numpy)
                free_mask = (np.abs(theta_to_grid - b) <= self.beta[i]/2.0 * self.free_bearing_inflate) & (dist_to_grid < (r - self.alpha/2.0)) & (dist_to_grid < self.z_max)
                occ_mask = (np.abs(theta_to_grid - b) <= self.beta[i]/2.0 * self.ocp_bearing_inflate) & (np.abs(dist_to_grid - r) <= self.alpha/2.0)

                # occ_mask_2 = dilation(occ_mask) & (np.invert(occ_mask))
                # Adjust the cells appropriately
                # self.state_map[occ_mask] = self.occupied_state
                # self.state_map[free_mask] = self.free_state
                self.log_prob_map[occ_mask] += self.l_occ
                # self.log_prob_map[occ_mask_2] += self.l_occ / 10
                self.log_prob_map[free_mask] += self.l_free

if __name__ == "__main__":
    yaml_setting = imp_setting("omp_settings.yaml")

    depth_plot = yaml_setting.plot_depth
    env_align = yaml_setting.manual_align

    folder = yaml_setting.data_dir
    work_folder = os.path.join(folder, "coarse_scan_images")
    img_folder = os.path.join(work_folder, "images")
    f_path = os.path.join(work_folder, "airsim_rec.txt")

    plot_dir = os.path.join(work_folder, "plot")
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    all_steps = []
    data = np.loadtxt(f_path, delimiter='\t', dtype=str)

    # coordinates
    # T_w_c = np.eye(4)
    # T_w_c[1, 1], T_w_c[2, 2] = -1, -1
    pose0 = Pose3(yaml_setting.tf_bodyInW)

    if env_align:
        env_align_file = os.path.join(folder, "align_line.txt")
        env_data_file = os.path.join(folder, "env_align.csv")
        slope = None
        if os.path.exists(env_align_file):
            slope, offset = np.loadtxt(fname=env_align_file)
        elif os.path.exists(env_data_file):
            line_data = np.loadtxt(fname=env_data_file, delimiter=",", dtype=float)
            offset, slope = P.polyfit(line_data[:, 0], line_data[:, 1], 1)
            np.savetxt(fname=env_align_file,X=np.array([slope, offset]))
        if slope is not None:
            angle = -np.arctan(slope)
            T_align = np.eye(4)
            T_align[:2, :2] = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            pose0 = Pose3(T_align) * pose0

    # depth images
    w, h, fov = yaml_setting.dep_w, yaml_setting.dep_h, yaml_setting.fov
    focal_len = w / 2 / np.tan(np.pi * fov/360)
    mid_r = int(h / 2)
    bearing_res = fov*np.pi/180/w
    beam_bearing, beam_width = ray_bearing_width(w, focal_len)
    px_y = np.arange(0, w)
    px_z = np.ones(w, dtype=int) * int(h / 2)
    beam_ranges = []

    # grid mapping
    mid_scans = []
    mid_segs = []
    mid_nmls = []
    grid_size = yaml_setting.grid_size
    depth_max = yaml_setting.depth_max
    free_dot_threshold = yaml_setting.free_dot_threshold
    grid_map = OccupMap(int(yaml_setting.grid_x_len / grid_size),
                        int(yaml_setting.grid_y_len / grid_size),
                        yaml_setting.grid_x_offset,
                        yaml_setting.grid_y_offset,
                        grid_size,
                        yaml_setting.obstacle_thickness,
                        beam_width, z_max=depth_max,
                        free_bearing_inflate=yaml_setting.free_bearing_inflate,
                        ocp_bearing_inflate=yaml_setting.ocp_bearing_inflate)

    # depth type
    dep_img_path = None
    rgb_img_path = None
    dep_type = "Planar"

    for r_id in tqdm(range(1, len(data))):
        row = data[r_id]
        # VehicleName	TimeStamp	POS_X	POS_Y	POS_Z	Q_W	Q_X	Q_Y	Q_Z	ImageFile
        T = [float(row[i]) for i in range(2, 9)]
        body_pose = pose0 * Pose3.by_vec(*T)
        paths = row[9].split(";")
        for img_path in paths:
            if img_path[:3] == "RGB":
                rgb_img_path = os.path.join(img_folder, img_path)
            elif img_path[:3] == "DEP":
                dep_img_path = os.path.join(img_folder, img_path)
        assert dep_img_path is not None and rgb_img_path is not None
        # depth image
        dep_img = read_pfm(dep_img_path)
        px_d = dep_img[mid_r, :]
        body_xyz = px2coor(w, h, px_y, px_z, px_d, focal_len, dep_type)
        body_xyz = np.hstack((body_xyz, np.ones((body_xyz.shape[0], 1))))
        # beam for occupancy grid mapping
        beam_range = np.linalg.norm(body_xyz[:, :2], axis=1)
        beam_ranges.append(beam_range)
        w_xyz = (body_pose.mat @ body_xyz.T).T[:, :3]
        mid_scans.append(w_xyz)
        if depth_plot:
            dep_img_plot_path = os.path.join(plot_dir, f"DEP{r_id-1}.png")
            if not os.path.exists(dep_img_plot_path):
                plt.imsave(fname=dep_img_plot_path, arr=dep_img, cmap=cm.Greys_r, vmax=yaml_setting.depth_max)

        arr0 = body_pose.tf_point(np.zeros(3))
        arr1 = body_pose.tf_point(np.array([4, 0, 0]))
        arr_pose = np.array([arr0[0], arr0[1], np.arctan2(arr1[1]-arr0[1], arr1[0]-arr0[0])])
        grid_map.update_map(arr_pose, np.hstack((beam_range.reshape((-1, 1)), beam_bearing.reshape((-1, 1)))))  # update the map

    fig1, ax1 = plt.subplots(figsize = (8, 8))
    ax1.pcolor(grid_map.grid_position_m[0], grid_map.grid_position_m[1],
               1.0 - 1. / (1. + np.exp(grid_map.log_prob_map)), cmap='Greys')
    free_space = ((1.0 - 1. / (1. + np.exp(grid_map.log_prob_map))) < free_dot_threshold)
    free_dot_x_idx, free_dot_y_idx = np.array(free_space).nonzero()
    # free_space = erosion(free_space)
    free_dots = np.array([grid_map.grid_position_m[0][free_space].flatten(), grid_map.grid_position_m[1][free_space].flatten()])
    true_free_dots = []
    scans = np.vstack(mid_scans)
    for i, dot in enumerate(free_dots.T):
        x, y = dot[:]

        if min(np.linalg.norm(dot - scans[:, :2], axis=1)) < yaml_setting.obstacle_thickness/2:
            # remove dots that are close to the scaned dots
            continue

        # remove isolated free dots
        x_idx, y_idx = free_dot_x_idx[i], free_dot_y_idx[i]
        connected = False

        if x_idx == 0:
            connected = connected or free_space[x_idx+1][y_idx]
        elif x_idx == grid_map.log_prob_map.shape[0] - 1:
            connected = connected or free_space[x_idx-1][y_idx]
        else:
            connected = connected or free_space[x_idx+1][y_idx] or free_space[x_idx-1][y_idx]

        if not connected:
            if y_idx == 0:
                connected = connected or free_space[x_idx][y_idx+1]
            elif y_idx == grid_map.log_prob_map.shape[1] - 1:
                connected = connected or free_space[x_idx][y_idx - 1]
            else:
                connected = connected or free_space[x_idx][y_idx - 1] or free_space[x_idx][y_idx+1]
        if connected:
            true_free_dots.append([x, y])
    true_free_dots = np.array(true_free_dots)
    ax1.scatter(true_free_dots[:, 0], true_free_dots[:, 1], s=5.0, c="blue")
    for i in tqdm(range(len(mid_scans))):
        dots = mid_scans[i]
        ax1.scatter(dots[:, 0], dots[:, 1], s=1.0, c="red")
    ax1.set_xlim(yaml_setting.grid_x_offset,
                 yaml_setting.grid_x_offset+yaml_setting.grid_x_len)
    ax1.set_ylim(yaml_setting.grid_y_offset,
                 yaml_setting.grid_y_offset+yaml_setting.grid_y_len)
    ax1.set_aspect('equal')
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    plt.show()
    fig1.savefig(fname=os.path.join(folder, "grid.png"), dpi=300)
    # unrotate points
    pose0_inv = pose0.inverse()
    tf_pts = pose0_inv.mat @ np.hstack([true_free_dots, np.zeros((len(true_free_dots), 1)), np.ones((len(true_free_dots), 1))]).T
    tf_pts = tf_pts[:2, :].T
    np.savetxt(X=tf_pts, fname=os.path.join(folder, "free_dot.txt"))