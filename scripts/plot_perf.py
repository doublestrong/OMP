import os
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils import imp_setting, Pose3

def compute_loc_rate(err_file_path, t_th, angle_th):
    data = np.loadtxt(err_file_path)
    mask = (data[:, 0] < t_th) & (data[:, 1] < angle_th)
    return np.sum(mask)/len(data)

if __name__ == "__main__":
    try:
        import cPickle as pickle
    except ModuleNotFoundError:
        import pickle
    plt.rcParams.update({'font.size': 16})
    params = imp_setting("omp_settings.yaml")
    use_result_folder = True
    label_tag_det = False
    res_folder = "result"
    np.random.seed(params.random_seed)
    random.seed(params.random_seed)
    eva_folder = os.path.join(params.data_dir, "evaluation")
    plot_folder = os.path.join(eva_folder, "figs")
    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)
    if use_result_folder:
        plot_folder = os.path.join(plot_folder, res_folder)
        if not os.path.exists(plot_folder):
            os.mkdir(plot_folder)

    no_marker_case = "NoMarker"
    if 0 in params.eva_marker_nums:
        case_names = [no_marker_case]
        labels = ["No marker"]
    else:
        case_names = []
        labels = []

    default_case = "Pct90"
    # default_case = "none"

    pct_names = [f"Pct{i}Marker" for i in params.marker_percentages]
    case_names += pct_names
    labels += [f"Pct{i}" if f"Pct{i}" != default_case else "Ours" for i in params.marker_percentages]
    # labels += [f"Ours ($v={i})$" for i in params.marker_percentages]


    rd_cases = [f"Rd{i}Marker" for i in params.eva_random_case_ids]
    case_names += rd_cases
    plot_random = False
    if len(rd_cases)>0:
        plot_random = True

    uc_cases = [f"Uc{i}Marker" for i in params.eva_unif_contour_case_ids]
    case_names += uc_cases
    plot_uc = False
    if len(uc_cases)>0:
        plot_uc = True


    bodyInW = Pose3(params.tf_bodyInW)
    camInB = Pose3(params.tf_camInB)

    xlim = [params.grid_x_offset, params.grid_x_offset + params.grid_x_len]
    ylim = [params.grid_y_offset, params.grid_y_offset + params.grid_y_len]

    test_cases = []
    titles = []
    for test_data_type in params.test_data_distributions:
        if test_data_type == "unif":
            title = "Uniform Test Data"
            test_file_name = f"test_{test_data_type}"
            titles.append(title)
            test_cases.append(test_file_name)
        else:
            for mean_scale in params.test_sample_mean_weight_scales:
                test_file_name = f"test_{test_data_type}_meanScale{mean_scale}"
                title = f"Nonuniform Test Data ({mean_scale})"
                titles.append(title)
                test_cases.append(test_file_name)

    tag_loc_fractions = params.tag_loc_percent

    metric_names = ["TEmedian", "REmedian"]
    loc_thresholds = [[0.005, 0.5], [0.01, 1], [0.05, 5], [0.1, 10], [1, 30]]

    metric_strs = ["Trans. Err. Median (m)", "Rot. Err. Median (DEG)"]
    metric_strs += [f"Recall ({100*line[0]} cm,{line[1]} deg) (%)" for line in loc_thresholds]
    # loc_strs = [f"Trans. Err.<{v} m" for v in scs_trans_max]
    # metric_str += [f"Rot. Err.<{v} DEG" for v in scs_angle_max]
    # metric_str += ["Loc. Rate (%)"] * len(additional_names)

    m_nums = params.eva_marker_nums
    markers = ['o','v','^','<','>','1','2','3','4','8','s','p','*','h','H','+','x','D','d']
    colors = []
    for i in range(len(markers)):
        colors.append('#%06X' % random.randint(0, 0xFFFFFF))
    m_size=10

    default_tag_loc = 1.0

    # dict hierarchy
    # test: case: tag: num: [perf,]
    all_res = {}
    # read in data
    for test_case in test_cases:
        all_res[test_case] = {}
        for c_i, c_name in enumerate(case_names):
            all_res[test_case][c_name] = {}
            for taglocfrac in tag_loc_fractions:
                if c_name == no_marker_case:
                    taglocfrac = 0
                    if taglocfrac in all_res[test_case][c_name]:
                        continue
                all_res[test_case][c_name][taglocfrac] = {'num':[],'perf':[]}
                for m_num in params.eva_marker_nums:
                    if m_num == 0 and c_name == no_marker_case:
                        case_folder = os.path.join(eva_folder, c_name)
                    elif m_num !=0 and c_name != no_marker_case:
                        case_folder = os.path.join(eva_folder, c_name, f"{m_num}markers")
                    else:
                        continue

                    err_file = os.path.join(case_folder, f'pkl/{res_folder}', f'test_err_{test_case}_{taglocfrac}tagloc.txt')

                    if os.path.exists(err_file):
                        err_res = np.loadtxt(err_file)
                        perfs = all_res[test_case][c_name][taglocfrac]['perf']
                        perf = []
                        perfs.append(perf)
                        nums = all_res[test_case][c_name][taglocfrac]['num']
                        nums.append(m_num)
                        for metric in metric_names:
                            if metric == "TEmedian":
                                perf.append(np.median(err_res[:,0]))
                            elif metric == "REmedian":
                                perf.append(np.median(err_res[:,1]))
                        for t_a in loc_thresholds:
                            perf += [100*np.sum((err_res[:,0] < t_a[0]) & (err_res[:,1] < t_a[1]) )/params.test_img_num for t_a in loc_thresholds]

    # plotting
    for t_i, test_case in enumerate(test_cases):
        for m_i, metric_str in enumerate(metric_strs):
            fig, ax = plt.subplots()
            has_no_marker = False
            lgd_idx = 0
            rd_num2perf = {}
            uc_num2perf = {}
            plot_no_marker = False
            for c_i, c_name in enumerate(case_names):
                for taglocfrac in tag_loc_fractions:
                    if c_name == no_marker_case and not plot_no_marker:
                        taglocfrac = 0
                        plot_no_marker = True
                    if taglocfrac in all_res[test_case][c_name]:
                        tmp_res = all_res[test_case][c_name][taglocfrac]
                        nums = tmp_res['num']
                        perfs = tmp_res['perf']
                        values = [perfs[i][m_i] for i in range(len(nums))]
                        if c_name[:2] == "Rd":
                            if taglocfrac not in rd_num2perf:
                                rd_num2perf[taglocfrac] = {}
                            for i, num in enumerate(nums):
                                if num not in rd_num2perf[taglocfrac]:
                                    rd_num2perf[taglocfrac][num] = [perfs[i][m_i]]
                                else:
                                    rd_num2perf[taglocfrac][num].append(perfs[i][m_i])
                        elif c_name[:2] == "Uc":
                            if taglocfrac not in uc_num2perf:
                                uc_num2perf[taglocfrac] = {}
                            for i, num in enumerate(nums):
                                if num not in uc_num2perf[taglocfrac]:
                                    uc_num2perf[taglocfrac][num] = [perfs[i][m_i]]
                                else:
                                    uc_num2perf[taglocfrac][num].append(perfs[i][m_i])
                        else:
                            if c_name==no_marker_case:
                                if not has_no_marker:
                                    ax.plot(nums, values, c=colors[lgd_idx], marker=markers[lgd_idx], markersize=m_size,
                                            label=labels[c_i], markerfacecolor='none', linestyle='')
                                    lgd_idx += 1
                                    has_no_marker = True
                            elif taglocfrac == default_tag_loc:
                                if label_tag_det:
                                    ax.plot(nums, values, c=colors[lgd_idx], marker=markers[lgd_idx], markersize=m_size,
                                            label=labels[c_i]+f" (TagDetectOn)", markerfacecolor='none', linestyle='')
                                else:
                                    ax.plot(nums, values, c=colors[lgd_idx], marker=markers[lgd_idx], markersize=m_size,
                                            label=labels[c_i], markerfacecolor='none', linestyle='')
                                lgd_idx += 1
                            else:
                                # ax.plot(nums, values, c=colors[lgd_idx], marker=markers[lgd_idx], markersize=m_size, label=labels[c_i]+f"(TagLoc{taglocfrac})",
                                #         linestyle='')
                                ax.plot(nums, values, c=colors[lgd_idx], marker=markers[lgd_idx], markersize=m_size, label=labels[c_i]+f" (TagDetectOff)",
                                        linestyle='')
                                lgd_idx +=1
                            np.savetxt(fname=os.path.join(plot_folder, f"{test_case}_{metric_str}_{labels[c_i]}.txt"),
                                       X=values)
            if len(rd_num2perf)> 0:
                for taglocfrac in rd_num2perf:
                    num2perf = rd_num2perf[taglocfrac]
                    nums = [num for num in num2perf]
                    means = [np.mean(num2perf[num]) for num in num2perf]
                    stds = [np.std(num2perf[num]) for num in num2perf]
                    if taglocfrac == default_tag_loc:
                        label = "Random"
                    else:
                        label = "Random"+f" (TagLoc{taglocfrac})"
                    ax.errorbar(nums, means, yerr=stds, xerr=None, marker=markers[lgd_idx],
                                mfc=colors[lgd_idx], mec=colors[lgd_idx], ms=m_size , label=label, linestyle='',ecolor=colors[lgd_idx])
                    lgd_idx += 1
                    np.savetxt(fname=os.path.join(plot_folder, f"{test_case}_{metric_str}_{label}.txt"), X=np.array([means, stds]))


            if len(uc_num2perf)> 0:
                for taglocfrac in uc_num2perf:
                    num2perf = uc_num2perf[taglocfrac]
                    nums = [num for num in num2perf]
                    means = [np.mean(num2perf[num]) for num in num2perf]
                    stds = [np.std(num2perf[num]) for num in num2perf]
                    if taglocfrac == default_tag_loc:
                        label = "Uniform"
                    else:
                        label = "Uniform"+f" (TagLoc{taglocfrac})"
                    ax.errorbar(nums, means, yerr=stds, xerr=None, marker=markers[lgd_idx],
                                mfc=colors[lgd_idx], mec=colors[lgd_idx], ms=m_size , label=label, linestyle='',ecolor=colors[lgd_idx])
                    lgd_idx += 1
                    np.savetxt(fname=os.path.join(plot_folder, f"{test_case}_{metric_str}_{label}.txt"), X=np.array([means, stds]))


            ax.set_xlabel("# of Markers")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # if metric_str[:6] == "Recall":
            #     ax.set_ylim([40,70])
            # ax.set_title(titles[t_i])
            if metric_str[:6] == "Recall":
                ax.set_yticks([50, 55, 60,65,70])
                # ax.set_yticks([71.0, 72.0, 73.0, 74.0, 75.0])
                # ax.set_yticks([85.0, 85.5, 86.0, 86.5, 87.0])
            ax.set_xticks(params.eva_marker_nums)
            ax.legend()
            ax.set_ylabel(metric_str)
            fig.savefig(os.path.join(plot_folder, f"{test_case}_{metric_str}.png"), bbox_inches='tight',dpi=300)