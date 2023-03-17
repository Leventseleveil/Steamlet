# -*- encoding: utf-8 -*-
# @Time: 2022/03/31 11:19
# @Author: librah
# @Description: L2A ABR algorithm(Online learning for low-latency adaptive streaming)
# @File: l2a.py
# @Version: 1.0

import numpy as np
import math

from config import Env_Config
from algorithms.abr import ABR_Solver

class L2A_Solver(ABR_Solver):
    def __init__(self):
        # General
        super(L2A_Solver, self).__init__()
        
        # L2A
        self.lastQuality = 0
        self.currentPlaybakcRate = 1.0
        self.prev_w = [0] * len(self.bitrates)
        self.w = [0] * len(self.bitrates)
        self.horizon = 4
        self.v1 = math.pow(self.horizon, 0.99)
        self.alpha = max(math.pow(self.horizon, 1), self.v1 * math.sqrt(self.horizon))
        self.Q = self.v1
        self.react = 2

        # For Dash playback rate
        self.LIVE_DELAY = 1.
        self.MIN_PLAYBACK_RATE_CHANGE = 0.02
        self.LIVE_CATCHUP_PLAYBACK_RATE = 0.1
    
    def reset(self):
        """
        Description: Reset the L2A Solver
        Args: None
        Return: None
        """
        self.tp_history = []
        self.latency_history = []
        self.prev_w = [0] * len(self.bitrates)
        self.w = [0] * len(self.bitrates)
        self.currentPlaybakcRate = 1.0
        self.v1 = math.pow(self.horizon, 0.99)
        self.alpha = max(math.pow(self.horizon, 1), self.v1 * math.sqrt(self.horizon))
        self.Q = self.v1
    
    def adjust_rate(self):
        """
        Description: Adjust the quality value and set prev_w
        Args: None
        Return: None
        """
        lastThroughput = self.harmonic_prediction()
        self.lastQuality = self.choose_rate(lastThroughput)
        self.prev_w[self.lastQuality] = 1
    
    def solve(self, buffer_length, curr_latency, player_state):
        # First of all, get speed
        ## DASH default playbac rate adaption
        speed, speed_idx = self.dash_playback_rate(curr_latency, buffer_length, player_state, self.currentPlaybakcRate)

        self.currentPlaybakcRate = speed

        # Get bitrate
        diff1 = []
        lastThroughput = self.harmonic_prediction()
        lastSegmentDuration = self.seg_duration / Env_Config.ms_in_s
        V = lastSegmentDuration
        sign = 1
        for i in range(len(self.bitrates)):
            if self.currentPlaybakcRate * self.bitrates[i] / Env_Config.kb_in_mb > lastThroughput:
                # In this case buffer would deplete, leading to a stall, which increases latency and thus the particular probability of selsection of bitrate[i] should be decreased.
                sign = -1
            # The objective of L2A is to minimize the overall latency=request-response time + buffer length after download+ potential stalling (if buffer less than chunk downlad time)
            self.w[i] = self.prev_w[i] + sign * (V /(2 * self.alpha)) * (self.Q + self.v1) * (self.currentPlaybakcRate * self.bitrates[i] / Env_Config.kb_in_mb / lastThroughput) #Lagrangian descent
        
        temp = [0] * len(self.bitrates)
        for i in range(len(self.bitrates)):
            temp[i] = abs(self.bitrates[i] - np.dot(self.w, self.bitrates))
        
        quality = temp.index(min(temp))
        # We employ a cautious -stepwise- ascent
        if quality > self.lastQuality:
            if self.bitrates[self.lastQuality + 1] / Env_Config.kb_in_mb <= lastThroughput:
                quality = self.lastQuality + 1

        # Provision against bitrate over-estimation, by re-calibrating the Lagrangian multiplier Q, to be taken into account for the next chunk
        if self.bitrates[quality] / Env_Config.kb_in_mb >= lastThroughput:
            self.Q = self.react * max(self.v1, self.Q)
        self.lastQuality = quality

        print("Best reward: quality--{}, speed--{}".format(self.bitrates[quality], self.speeds[speed_idx]))

        return quality, speed_idx