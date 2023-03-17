# -*- encoding: utf-8 -*-
# @Time: 2022/03/30 15:40
# @Author: librah
# @Description: Live Video Testing for all algorithms
# @File: testing.py
# @Version: 1.0

import argparse
import os

import emulator as Env
from config import Config, Env_Config
from utils import get_tp_time_trace_info
from algorithms.lol import LoL_Solver
from algorithms.l2a import L2A_Solver
from algorithms.stallion import Stallion_Solver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', dest='abr_algorithm', help='ABR algorithm', default='lol', type=str)

    parser.add_argument('-m', '--massive', dest='massive', help='massive testing',
                        default=False, action='store_true', required=False)

    parser.add_argument('-r', '--random', dest='random_latency',
                        help='use random latency', default=False, action='store_true', required=False)

    parser.add_argument('-l', '--latency', dest='init_latency',
                        help='initial latency', default=2, type=int, required=False)

    parser.add_argument('-b', '--bw_amplify', dest='bw_amplify', help='amplify bandwidth', default=False, action='store_true')

    args = parser.parse_args()
    return args

def choose_abr_solver(algo):
    """
    Description: Choose the matched algorithm
    Args:
        algo: abr algorithm name
    Return:
        solver: the matched abr algorithm
    """
    algo_dict = {
        'lol': LoL_Solver(),
        'l2a': L2A_Solver(),
        'stallion': Stallion_Solver()
    }

    return algo_dict.get(algo)

def test(args):
    algo = args.abr_algorithm
    massive = args.massive
    init_latency = args.init_latency
    random_latency = args.random_latency
    bw_amplify = args.bw_amplify

    env = Env.Live_Streaming(initial_latency=init_latency, testing=True, massive=massive, random_latency=random_latency)
    _, action_dims = env.get_action_info()

    if massive:
        if bw_amplify:
            compare_path = Config.a_cdf_dir
            result_path = Config.a_massive_result_files + algo + '/latency_Nones/'
        else:
            compare_path = Config.cdf_dir
            result_path = Config.massive_result_files + algo + '/latency_Nones/'
        if not os.path.exists(compare_path):
            os.makedirs(compare_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        if random_latency:
            compare_file = open(compare_path + algo + '.txt', 'w')
        else:
            compare_file = open(compare_path + algo + str(int(init_latency)) + 's.txt', 'w')
        
        # Choose the matched algorithm
        solver = choose_abr_solver(algo)
        # print(abr_solver.__class__)
        
        while True:
            # Start testing
            env_end = env.reset(testing=True, bw_amplify=bw_amplify)
            solver.reset()
            if env_end:
                break

            testing_start_time = env.get_server_time()
            print("Initial latency is: {}".format(testing_start_time))
            tp_trace, time_trace, trace_name, starting_idx = env.get_player_trace_info()
            print("Trace name is: {}".format(trace_name))

            log_path = result_path + trace_name
            log_file = open(log_path, 'w')
            # Default action value(bitrate: 0, speed: 1)
            avg_bw, reward = env.act(0, 1, massive=massive)
            latency = env.get_latency() / Env_Config.ms_in_s
            solver.update_tp_latency(avg_bw, latency)
            total_reward = 0.0
            while not env.streaming_finish():
                player_state = env.get_player_state()
                if player_state == 0:
                    # Set default value when the state is start up
                    action_1 = 0
                    action_2 = 1
                else:
                    tmp_buffer = env.get_buffer_length() / Env_Config.ms_in_s
                    tmp_latency = env.get_latency() / Env_Config.ms_in_s
                    action_1, action_2 = solver.solve(tmp_buffer, tmp_latency, player_state)
                avg_bw, reward = env.act(action_1, action_2, log_file, massive)
                latency = env.get_latency() / Env_Config.ms_in_s
                solver.update_tp_latency(avg_bw, latency)

                state_new = env.get_state()
                state = state_new
                total_reward += reward
                print("action_1:{}, action_2:{}, reward:{}".format(action_1, action_2, reward))
            print("File:{}, reward:{}, init latency:{}".format(trace_name, total_reward, testing_start_time))
            # Get initial latency of player and how long time is used. and tp/time trace
            testing_duration = env.get_server_time() - testing_start_time
            tp_record, time_record = get_tp_time_trace_info(tp_trace, time_trace, starting_idx, testing_duration + env.player.get_buffer())
            log_file.write('\t'.join(str(tp) for tp in tp_record))
            log_file.write('\n')
            log_file.write('\t'.join(str(time) for time in time_record))
            log_file.write('\n' + str(testing_start_time))
            log_file.write('\n')
            log_file.close()
            env.massive_save(trace_name, compare_file)

        compare_file.close()

if __name__ == '__main__':
    args = parse_args()
    test(args)