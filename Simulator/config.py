# -*- encoding: utf-8 -*-
# @Time: 2022/03/23 20:37
# @Author: librah
# @Description: Configuration for all files
# @File: config.py
# @Version: 1.0

class Config(object):
    # The global configuration set
    random_seed = 11
    massive_result_files = './all_results/'
    a_massive_result_files = './amplified_all_results/'
    regular_test_files = './debug/'
    a_regular_test_files = './amplified_debug/'
    cdf_dir = '../result_show/results/'
    a_cdf_dir = '../result_show/amplified_results/'
    trace_idx = 10


class Env_Config(object):
    # For environment, ms
    bw_env_version = 0  # O for LTE (NYC), 1 for 3G (Norway)
    if bw_env_version == 0:
        data_dir = '../bw_traces/'
        test_data_dir = '../bw_traces_test/cooked_test_traces/'
    elif bw_env_version == 1:
        data_dir = '../new_traces/train_sim_traces/'
        test_data_dir = '../new_traces/test_sim_traces/'
    # live video streaming session info
    s_info = 10
    s_len = 15
    a_num = 2
    a_dims = [6, 3]  # 6 bitrates and 3 playing speed
    video_terminal_length = 300  # 200 for training, 100 for testing
    # network and chunk info
    packet_payload_portion = 0.973
    rtt_low = 30.0
    rtt_high = 40.0
    range_low = 40
    range_high = 50
    chunk_random_ratio_low = 0.95
    chunk_random_ratio_high = 1.05

    # 6 video bitrates
    bitrate = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
    # 3 video playing speed
    speeds = [0.90, 1.0, 1.10]
    # video parameters
    ms_in_s = 1000.0
    kb_in_mb = 1000.0   # in ms
    seg_duration = 1000.0
    chunk_duration = 200.0
    chunk_in_seg = seg_duration/chunk_duration
    chunk_seg_ratio = chunk_duration/seg_duration
    server_init_lat_low = 3
    server_init_lat_high = 5
    start_up_ssh = 2000.0
    freezing_tol = 3000.0
    buffer_ub = server_init_lat_high*seg_duration

    default_action_1 = 0
    default_action_2 = 1
    skip_segs = 2.0
    repeat_segs = 2.0

    # Server encoding info
    bitrate_low_noise = 0.7
    bitrate_high_noise = 1.3
    ratio_low_2 = 2.0  # this is the lowest ratio between first chunk and the sum of all others
    # this is the highest ratio between first chunk and the sum of all others
    ratio_high_2 = 10.0
    ratio_low_5 = 0.75  # this is the lowest ratio between first chunk and the sum of all others
    # this is the highest ratio between first chunk and the sum of all others
    ratio_high_5 = 1.0
    est_low_noise = 0.98
    est_high_noise = 1.02

    # Reward metrics parameters
    action_reward = 1.5 * chunk_seg_ratio
    rebuf_penalty = 6.0
    smooth_penalty = 1.0
    long_delay_penalty_new = 0.1 * chunk_seg_ratio
    long_delay_penalty = 4.0 * chunk_seg_ratio
    const = 6.0
    x_ratio = 1.0
    speed_smooth_penalty = 2.0
    unnormal_playing_penalty = 2.0
    skip_seg_penalty = 2.0
    repeat_seg_penalty = 2.0
    skip_latency = skip_segs * seg_duration + chunk_duration


class Plot_Config(object):
    result_dir = './debug/'
    figures_dir = './test_figures/'
    result_file = './test_figures/'
    plt_buffer_a = 1e-5
