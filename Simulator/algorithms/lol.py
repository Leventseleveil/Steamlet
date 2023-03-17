# -*- encoding: utf-8 -*-
# @Time: 2022/03/30 19:48
# @Author: librah
# @Description: LOL ABR algorithm(When They Go High, We Go Low:  Low-Latency Live Streaming in dash.js with LoL)
# @File: lol.py
# @Version: 1.0

import math
import numpy as np

from itertools import product
from config import Env_Config
from algorithms.abr import ABR_Solver

class LoL_Solver(ABR_Solver):
    def __init__(self):
        # General
        super(LoL_Solver, self).__init__()
        self.eta = 0.9
        
        # Lol 
        self.lookahead = 3
        self.options = self.getProduct(Env_Config.bitrate)
        self.last_rate = self.bitrates[0] / Env_Config.kb_in_mb
        self.last_speed = 1.0
    
    def reset(self):
        """
        Description: Reset the LoL Solver
        Args: None
        Return: None
        """
        self.tp_history = []
        self.latency_history = []
        self.last_rate = self.bitrates[0] / Env_Config.kb_in_mb
        self.last_speed = 1.0

    def getProduct(self, arr):
        """
        Description: Get producr result of the arr input
        Args: 
            arr: input array
        Return:
            res: product result
        """
        # own method
        # res = [[]]
        # for i in range(self.lookahead):
        #     new_res = []
        #     for pre in res:
        #         for j in range(len(arr)):
        #             new_res.append(pre + [arr[j]])
        #     res = new_res

        # itertools method
        res = list(product(arr, repeat=self.lookahead))
        
        return res
    
    def solve(self, buffer_length, curr_latency, player_state):
        """
        Description: search for the right bitrate and speed when reward is best
        Args: 
            buffer_length: in s
            curr_latency: in s
            player_state: 0(start up), 1(traceing) 2(rebuffering)
        Return:
            bitrate_index: the index of bitrate result
            speed_index: the index of speed result
        """
        maxReward = float('-inf')
        bestOption = []
        bestQoeInfo = {}

        # Qoe stuff
        qoeEvaluatorTmp = QoeEvaluator()
        minBitrateMbps = self.bitrates[0] / Env_Config.kb_in_mb # in Mbps
        maxBitrateMbps = self.bitrates[-1] / Env_Config.kb_in_mb # in Mbps

        qualityList = []

        # Iterate over all options
        for i in range(len(self.options)):
            tmpLatency = curr_latency
            tmpBuffer = buffer_length
            curr_option = self.options[i]
            qoeEvaluatorTmp.setupPerSegmentQoe(self.seg_duration, maxBitrateMbps, minBitrateMbps, self.last_rate, self.last_speed)

            tmpSpeeds = []
            # Estimate futureBandwidth as harmonic mean of past X throughput values
            futureBandwidthMbps = self.eta * self.harmonic_prediction()

            # For each segment in lookahead window (window size: futureSegmentCount)
            for j in range(self.lookahead):
                segmentBitrateMbps = curr_option[j] / Env_Config.kb_in_mb

                futureSegmentSizeMbits = self.seg_duration * segmentBitrateMbps
                downloadTime = futureSegmentSizeMbits / futureBandwidthMbps
                print("downloadTime: {}".format(downloadTime))
                if downloadTime > tmpBuffer:
                    # buffer underflow, so as to rebuffer
                    segmentRebufferTime = downloadTime - tmpBuffer
                    tmpBuffer = self.chunk_duration
                    tmpLatency += segmentRebufferTime
                    player_state = 0
                else:
                    segmentRebufferTime = 0
                    tmpBuffer -= downloadTime
                    tmpBuffer += self.seg_duration
                    player_state = 1

                # Check if need to catch up
                if abs(tmpLatency - self.LIVE_DELAY) >= self.LIVE_CATCHUP_MIN_DRIFT:
                    needToCatchUp = True
                else:
                    needToCatchUp = False
                
                # If need to catch up, calculate new playback rate (custom/default methods)
                if needToCatchUp:
                    newRate, _ = self.dash_playback_rate(tmpLatency, tmpBuffer, player_state, self.last_speed)
                    futurePlaybackSpeed = newRate
                else:
                    futurePlaybackSpeed = 1.0

                tmpSpeeds += [futurePlaybackSpeed]
                catchupDuration = self.seg_duration - (self.seg_duration / futurePlaybackSpeed)
                futureLatency = tmpLatency - catchupDuration

                qoeEvaluatorTmp.updateQoe(segmentBitrateMbps, segmentRebufferTime, futureLatency, futurePlaybackSpeed)

                tmpLatency = futureLatency

            reward = qoeEvaluatorTmp.getPerSegmentQoe()

            if (reward > maxReward):
                maxReward = reward
                bestOption = curr_option
                bestSpeed = tmpSpeeds
        
        print("Best reward: option--{}, speed--{}".format(bestOption, bestSpeed))
        self.last_rate = bestOption[0]
        self.last_speed = bestSpeed[0]
        return self.bitrates.index(bestOption[0]), self.speeds.index(bestSpeed[0])

class QoeEvaluator(object):
    def __init__(self):
        self.voPerSegmentQoeInfo = None
        self.bitrates = Env_Config.bitrate
    
    def getPerSegmentQoe(self):
        return self.voPerSegmentQoeInfo.getTotalQoe()

    def updateQoe(self, segmentBitrateMbps, segmentRebufferTime, latency, playbackSpeed):
        """
        Description: Update the QoE value
        Args: 
            segmentBitrateMbps: bitrate for current segment(in Mbps)
            segmentRebufferTime: current rebuffer time(in s)
            latency: current latency(in s)
            playbackSpeed: playback speed
        Return: None
        """
        # bitrate reward
        log_quality = np.log(segmentBitrateMbps / (self.bitrates[0] / Env_Config.kb_in_mb))
        self.voPerSegmentQoeInfo.bitrateWSum += self.voPerSegmentQoeInfo.weights['bitrateReward'] * log_quality

        # bitrate switch penalty
        if self.voPerSegmentQoeInfo.lastBitrate:
            log_pre_quality = np.log(self.voPerSegmentQoeInfo.lastBitrate / (self.bitrates[0] / Env_Config.kb_in_mb))
            self.voPerSegmentQoeInfo.bitrateSwitchWSum += self.voPerSegmentQoeInfo.weights['bitrateSwitchPenalty'] * abs(log_quality - log_pre_quality)
        self.voPerSegmentQoeInfo.lastBitrate = segmentBitrateMbps

        # rebuffer penalty
        self.voPerSegmentQoeInfo.rebufferWSum += self.voPerSegmentQoeInfo.weights['rebufferPenalty'] * segmentRebufferTime

        # latency penalty
        self.voPerSegmentQoeInfo.latencyWSum += self.voPerSegmentQoeInfo.weights['latencyPenalty'] * latency

        # palyback speed penalty
        self.voPerSegmentQoeInfo.playbackSpeedWSum += self.voPerSegmentQoeInfo.weights['playbackSpeedPenalty'] * abs(1 - playbackSpeed)

        # speed change penalty
        if self.voPerSegmentQoeInfo.lastSpeed:
            lastSpeed = self.voPerSegmentQoeInfo.lastSpeed
            self.voPerSegmentQoeInfo.speedChangeWSum += self.voPerSegmentQoeInfo.weights['speedChangePenalty'] * abs(playbackSpeed - lastSpeed)
        self.voPerSegmentQoeInfo.lastSpeed = playbackSpeed

        # update Total Qoe Value
        self.updateTotalQoe()

    def setupPerSegmentQoe(self, segmentDuration, maxBitrateMbps, minBitrateMbps, lastRate, lastSpeed):
        # Set up Per Segment QoeInfo
        self.voPerSegmentQoeInfo = self.createQoeInfo('segment', segmentDuration, maxBitrateMbps, minBitrateMbps)
        self.setInitialLastBitrate(lastRate)
        self.setInitialLastSpeed(lastSpeed)
    
    def setInitialLastBitrate(self, lastRate):
        self.voPerSegmentQoeInfo.lastBitrate = lastRate

    def setInitialLastSpeed(self, lastSpeed):
        self.voPerSegmentQoeInfo.lastSpeed = lastSpeed

    def updateTotalQoe(self):
        return self.voPerSegmentQoeInfo.updateTotalQoe()
    
    def createQoeInfo(self, fragmentType, fragmentDuration, maxBitrateMbps, minBitrateMbps):
        """
        Description: Create the QoeInfo object
                    * [Weights][Source: Abdelhak Bentaleb, 2020 (last updated: 30 Mar 2020)]
                    * bitrateReward: segment duration, e.g. 0.5s
                    * bitrateSwitchPenalty: 0.02s or 1s if the bitrate switch is too important
                    * rebufferPenalty: max encoding bitrate, e.g. 1 Mbps
                    * latencyPenalty: if L ≤ 1.1 seconds then = min encoding bitrate * 0.05, otherwise = max encoding bitrate * 0.1
                    * playbackSpeedPenalty: min encoding bitrate, e.g. 0.2 Mbps
        Args: 
            fragmentType: the type of fragment
            fragmentDuration: the duration of fragment
            maxBitrateMbps: the maximum bitrate(in Mbps)
            minBitrateMbps: the minimum bitrate(in Mbps)
        Return:
            qoeInfo: QoeInfo object
        """
        # Create new QoeInfo object
        qoeInfo = QoeInfo()
        qoeInfo.QoeType = fragmentType

        # Set weight: bitrateReward
        if not fragmentDuration:
            qoeInfo.weights['bitrateReward'] = 0.001
        else:
            qoeInfo.weights['bitrateReward'] = Env_Config.action_reward * Env_Config.chunk_in_seg
        
        # Set weight: bitrateSwitchPenalty
        qoeInfo.weights['bitrateSwitchPenalty'] = Env_Config.smooth_penalty

        # Set weight: rebufferPenalty
        if not maxBitrateMbps:
            qoeInfo.weights['rebufferPenalty'] = Env_Config.rebuf_penalty
        else:
            qoeInfo.weights['rebufferPenalty'] = Env_Config.rebuf_penalty
        
        # Set weight: latencyPenalty
        qoeInfo.weights['latencyPenalty'] = Env_Config.long_delay_penalty_new*Env_Config.chunk_in_seg
        # qoeInfo.weights[latencyPenalty].append({threshold: 1.1, penalty: 0.5*long_delay_penalty_new*chunk_in_seg})
        # qoeInfo.weights[latencyPenalty].append({threshold: float('inf'), penalty: long_delay_penalty_new*chunk_in_seg})

        # Set weight: playbackSpeedPenalty
        if not minBitrateMbps:
            qoeInfo.weights['playbackSpeedPenalty'] = Env_Config.unnormal_playing_penalty
        else: 
            qoeInfo.weights['playbackSpeedPenalty'] = Env_Config.unnormal_playing_penalty
        
        qoeInfo.weights['speedChangePenalty'] = Env_Config.speed_smooth_penalty

        return qoeInfo

class QoeInfo(object):
    def __init__(self):
        self.QoeType = None
        self.lastBitrate = None
        self.lastSpeed = None
        
        self.weights = dict()
        self.weights['bitrateReward'] = None
        self.weights['bitrateSwitchPenalty'] = None
        self.weights['rebufferPenalty'] = None
        self.weights['latencyPenalty'] = None
        self.weights['playbackSpeedPenalty'] = None
        self.weights['speedChangePenalty'] = None

        self.bitrateWSum = 0
        self.bitrateSwitchWSum = 0     # kbps
        self.rebufferWSum = 0          # seconds
        self.latencyWSum = 0           # seconds
        self.playbackSpeedWSum = 0     # e.g. 0.95, 1.0, 1.05
        self.speedChangeWSum = 0

        self.totalQoe = 0
    
    def getTotalQoe(self):
        return self.totalQoe
    
    def updateTotalQoe(self):
        self.totalQoe = self.bitrateWSum - self.bitrateSwitchWSum - self.rebufferWSum - self.latencyWSum - self.playbackSpeedWSum - self.speedChangeWSum