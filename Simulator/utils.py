# -*- encoding: utf-8 -*-
# @Time: 2022/03/23 21:04
# @Author: librah
# @Description: some utility functions that used to calculate network info
# @File: utils.py
# @Version: 1.0

import os
from config import Env_Config


def load_bandwidth(testing=False):
    """
    Description: load bandwidth info from network trace files
    Args:
        testing: True/False(choose the test trace file or not)
    Return:
        time_traces: trace throughput time list
        throughput_traces: trace throughput list
        data_names: trace data list
    """
    if testing:
        data_dir = Env_Config.test_data_dir
        datas = os.listdir(Env_Config.test_data_dir)
    else:
        data_dir = Env_Config.data_dir
        datas = os.listdir(Env_Config.data_dir)

    time_traces = []
    throughput_traces = []
    data_names = []

    if Env_Config.bw_env_version == 0:
        # for LTE trace
        for data in datas:
            if '.DS' in data:
                continue
            file_path = os.path.join(data_dir, data)
            time_trace = []
            throughput_trace = []
            time = 0.0
            with open(file_path, 'r') as f:
                for line in f:
                    parse = line.strip('\n')
                    time_trace.append(time)
                    throughput_trace.append(float(parse))
                    time += 1.0
            time_traces.append(time_trace)
            throughput_traces.append(throughput_trace)
            data_names.append(data)
    elif Env_Config.bw_env_version == 1:
        for data in datas:
            if '.DS' in data:
                continue
            file_path = os.path.join(data_dir, data)
            time_trace = []
            throughput_trace = []
            with open(file_path, 'r') as f:
                for line in f:
                    parse = line.strip('\n').split()
                    time_trace.append(float(parse[0]))
                    throughput_trace.append(float(parse[1]))
            time_traces.append(time_trace)
            throughput_traces.append(throughput_trace)
            data_names.append(data)
    return time_traces, throughput_traces, data_names


def load_single_trace():
    """
    Description: load bandwidth info from single network trace file
    Return:
        time_trace throughput_trace
    """
    file_path = Env_Config.data_dir
    time_trace = []
    throughput_trace = []
    time = 0.0

    with open(file_path, 'r') as f:
        if Env_Config.bw_env_version == 0:
            for line in f:
                parse = line.strip('\n')
                time_trace.append(time)
                throughput_trace.append(float(parse))
                time += 1.0
        elif Env_Config.bw_env_version == 1:
            for line in f:
                parse = line.strip('\n').split()
                time_trace.append(float(parse[0]))
                throughput_trace.append(float(parse[1]))
    return time_trace, throughput_trace


def get_tp_time_trace_info(tp_trace, time_trace, starting_time_idx, duration):
    """
    Description: get the throughput and time record that given start_time and duration
    Args:
        tp_trace: throughput list from one trace
        time_trace: time list from one trace
        starting_time_idx: start time index
        duration: time duration(ms)
    Return:
        tp_record: throughput record list
        time_record: time record list
    """
    start_time = time_trace[starting_time_idx]
    tp_record = []
    time_record = []
    offset = 0
    time_offset = 0.0
    i = 0
    time_range = 0.0
    while time_range < duration/Env_Config.ms_in_s:
        time_index = starting_time_idx + i + offset
        tp_record.append(tp_trace[time_index])
        time_record.append(time_trace[time_index] + time_offset)
        i += 1
        # repeat the throughput trace
        if starting_time_idx + i + offset >= len(tp_trace):
            offset -= len(tp_trace)
            time_offset += time_trace[-1]
        time_range = time_trace[starting_time_idx + i + offset] + time_offset - start_time

    return tp_record, time_record
