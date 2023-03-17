# -*- encoding: utf-8 -*-
# @Time: 2022/04/01 16:13
# @Author: librah
# @Description: STALLION ABR algorithm(STALLION: Video Adaptation Algorithm for Low-Latency Video Streaming)
# @File: stallion.py
# @Version: 1.0

import numpy as np

from algorithms.abr import ABR_Solver

class Stallion_Solver(ABR_Solver):
    def __init__(self):
        # General
        super(Stallion_Solver, self).__init__()

        # Stallion for new traces
        self.tp_f = 1.0
        self.latency_f = 1.25
        self.target_latency = 2.0
        self.speed_buffer_tth = 0.6

    def solve(self, buffer_length, curr_latency):
        # First of all, get speed
        bitrate_index, speed_index = None, None
        if curr_latency >= self.target_latency and buffer_length >= self.speed_buffer_tth:
            speed_index = 2
        else:
            speed_index = 1

        # Get rate
        mean_tp, mean_latency = np.mean(self.tp_history), np.mean(self.latency_history)
        std_tp, std_latency = np.std(self.tp_history), np.std(self.latency_history)
        predict_tp = mean_tp - self.tp_f * std_tp
        predict_latency = mean_latency + self.latency_f * std_latency
        overhead = max(predict_latency - self.target_latency, 0)

        if overhead >= self.seg_duration:
            bitrate_index = 0
        else:
            dead_time = self.seg_duration - overhead
            ratio = dead_time / self.seg_duration
            predict_tp *= ratio
            bitrate_index = self.choose_rate(predict_tp)

        print("Best reward: quality--{}, speed--{}".format(self.bitrates[bitrate_index], self.speeds[speed_index]))

        return bitrate_index, speed_index