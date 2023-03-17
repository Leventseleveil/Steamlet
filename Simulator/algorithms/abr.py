# -*- encoding: utf-8 -*-
# @Time: 2022/03/31 16:24
# @Author: librah
# @Description: ABR algorithm base class
# @File: abr.py
# @Version: 1.0

import math
import numpy as np

from config import Env_Config

class ABR_Solver(object):
    def __init__(self):
        # General
        self.n_step = 10
        self.hm_pre_steps = 5

        self.tp_history = []
        self.latency_history = []
        self.seg_duration = Env_Config.seg_duration / Env_Config.ms_in_s
        self.chunk_duration = Env_Config.chunk_duration / Env_Config.ms_in_s
        self.bitrates = Env_Config.bitrate
        self.speeds = Env_Config.speeds

        # dash
        self.LIVE_DELAY = 1.5
        self.MIN_PLAYBACK_RATE_CHANGE = 0.05
        self.LIVE_CATCHUP_PLAYBACK_RATE = 0.1
        self.LIVE_CATCHUP_MIN_DRIFT = 0.1
    
    def reset(self):
        """
        Description: Reset the ABR Solver
        Args: None
        Return: None
        """
        self.tp_history = []
        self.latency_history = []
    
    def update_tp_latency(self, tp, latency):
        """
        Description: Add the throughput and latency record, while limit the lens to n_step.
        Args: 
            tp: the average bandwidth
            latency: the env's current latency
        Return: None
        """
        self.tp_history += [tp]
        self.latency_history += [latency]
        if len(self.tp_history) > self.n_step:
            self.tp_history.pop(0)
        if len(self.latency_history) > self.n_step:
            self.latency_history.pop(0)
    
    def harmonic_prediction(self):
        """
        Description: Get the future throughput according to harmonic prediction.
        Args: None
        Return: future throughput prediction
        """
        if len(self.tp_history) < self.hm_pre_steps:
            tmp = self.tp_history
        else:
            tmp = self.tp_history[-self.hm_pre_steps:]
        
        return len(tmp)/(np.sum([1/tp for tp in tmp]))
    
    def dash_playback_rate(self, curr_latency, buffer_length, player_state, last_speed):
        """
        Description: Choose the right dash playback rate
        Args: 
            curr_latency: current latency (in s)
            buffer_length: current buffer level (in s)
            player state: 0(start up), 1(traceing) 2(rebuffering)
            last_speed: last speed action
        Return:
            speed: the right dash playback rate
            speed_idx: speed index
        """
        cpr = self.LIVE_CATCHUP_PLAYBACK_RATE
        delta_latency = curr_latency - self.LIVE_DELAY
        d = delta_latency * 5
        # Playback rate must be between (1 - cpr) - (1 + cpr)
        # ex: if cpr is 0.5, it can have values between 0.5 - 1.5
        s = (cpr * 2) / (1 + math.pow(np.e, -d))
        speed = (1 - cpr) + s
        # take into account situations in which there are buffer stalls,
        # in which increasing playbackRate to reach target latency will
        # just cause more and more stall situations
        if player_state == 0 or player_state == 2:
            if buffer_length > self.LIVE_DELAY / 2:
                pass
            elif delta_latency > 0:
                speed = 1.0
        
        # don't change playbackrate for small variations (don't overload element with playbackrate changes)
        if abs(last_speed - speed) <= self.MIN_PLAYBACK_RATE_CHANGE:
            speed = last_speed
        
        # Change speed to index
        speed_idx = self.choose_speed(speed)

        return self.speeds[speed_idx], speed_idx
    
    def choose_speed(self, speed):
        """
        Description: Choose the closest speed index
        Args: 
            speed: speed action value
        Return:
            min_idx: the closest speed index
        """
        min_abs, min_idx = float('inf'), None
        for i in range(len(self.speeds)):
            if abs(self.speeds[i] - speed) < min_abs:
                min_abs = abs(self.speeds[i] - speed)
                min_idx = i
        
        return min_idx
    
    def choose_rate(self, tp):
        """
        Description: Choose the closest bitrate index
        Args: 
            tp: throughput value that predicted
        Return:
            i: the closest bitrate index
        """
        i = 0
        for i in reversed(range(len(self.bitrates))):
            if self.bitrates[i] / Env_Config.kb_in_mb < tp:
                return i
        
        return i
    
    def solve(self, buffer_length, curr_latency):
        """
        Description: search for the right bitrate and speed when reward is best
        Args: 
            buffer_length: in s
            curr_latency: in s
        Return:
            bitrate_index: the index of bitrate result
            speed_index: the index of speed result
        """
        pass