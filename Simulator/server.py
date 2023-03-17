# -*- encoding: utf-8 -*-
# @Time: 2022/03/23 22:29
# @Author: librah
# @Description: Live video streaming server
# @File: server.py
# @Version: 1.0

import numpy as np
from config import Config, Env_Config
from random import Random

class Live_Server(object):
    def __init__(self, initial_latency, random_seed=Config.random_seed):
        self.myRandom = Random(random_seed)
        self.latency_random = Random(random_seed+1)
        self.initial_latency = initial_latency
        self.seg_duration = Env_Config.seg_duration
        self.chunk_duration = Env_Config.chunk_duration
        self.chunk_in_seg = Env_Config.chunk_in_seg
        self.next_delivery = []  # The next delivery info [segment_index, chunk_start_index, chunk_end_index, chunks_size(for all bitrates)]

        self.time = (self.myRandom.random() + self.initial_latency)*Env_Config.seg_duration  # The current runtime
        self.current_seg_idx = -1  # FOr initial
        self.current_chunk_idx = 0
        self.chunks = []  # [[current_seg_idx, current_chunk_idx, current_chunk_size(for all bitrates), current_segment_size(for the first chunk all bitrates, others not)]]...]
        self.current_seg_size = [[] for i in range(len(Env_Config.bitrate))]
    
    def generate_chunk_size(self):
        """
        Description: Generate the chunk size for next/current segment(for all bitrates)
        Args:
            None
        Return:
            self.current_seg_size: [chunks_size(for all bitrates)]
        """
        self.current_seg_size = [[] for i in range(len(Env_Config.bitrate))]
        # Initial coef, all Env_Config.bitrate share the same coef 
        encoding_coef = self.myRandom.uniform(Env_Config.bitrate_low_noise, Env_Config.bitrate_high_noise)
        # get the segment size that estimated
        estimate_seg_size = [b * encoding_coef for b in Env_Config.bitrate]
        # There is still noise for prediction, all Env_Config.bitrate cannot share the coef exactly same
        seg_size = [self.myRandom.uniform(Env_Config.est_low_noise*e, Env_Config().est_high_noise*e) for e in estimate_seg_size]

        if self.chunk_in_seg == 2:
            # Distribute size for chunks, currently, it should depend on chunk duration (200 or 500)
            ratio = self.myRandom.uniform(Env_Config.ratio_low_2, Env_Config.ratio_high_2)
            seg_ratio = [self.myRandom.uniform(Env_Config.est_low_noise*ratio, Env_Config.est_high_noise*ratio) for i in range(len(Env_Config.bitrate))]

            for i in range(len(seg_ratio)):
                temp_ratio = seg_ratio[i]
                temp_aux_chunk_size = seg_size[i] / (1+temp_ratio)
                temp_ini_chunk_size = seg_size[i] - temp_aux_chunk_size
                self.current_seg_size[i].extend(temp_ini_chunk_size, temp_aux_chunk_size)
        # if 200ms, needs to be modified, not working
        elif self.chunk_in_seg == 5:
            # assert 1 == 0
            ratio = self.myRandom.uniform(Env_Config.ratio_low_5, Env_Config.ratio_high_5)
            seg_ratio = [self.myRandom.uniform(Env_Config.est_low_noise*ratio, Env_Config.est_high_noise*ratio) for i in range(len(Env_Config.bitrate))]

            for i in range(len(seg_ratio)):
                temp_ratio = seg_ratio[i]
                temp_ini_chunk_size = seg_size[i] * temp_ratio / (1+temp_ratio)
                temp_aux_chunk_size = (seg_size[i] - temp_ini_chunk_size) / (self.chunk_in_seg - 1)
                temp_chunks_size = [temp_ini_chunk_size]
                temp_chunks_size.extend([temp_aux_chunk_size for _ in range(int(self.chunk_in_seg) - 1)])
                self.current_seg_size[i].extend(temp_chunks_size)
    
    def encoding_update(self, starting_time, ending_time):
        """
        Description: The encoding update process
        Args: 
            starting_time: the start time for encoding(ms)
            ending_time: the end time for encoding(ms)
        Return:
            self.chunks: [[current_seg_idx, current_chunk_idx, current_chunk_size(for all bitrates), current_segment_size(for the first chunk all bitrates, others not)]]...]
        """
        temp_time = starting_time
        while True:
            # The time of next chunk
            next_time = (int(temp_time/self.chunk_duration) + 1) * self.chunk_duration
            if next_time > ending_time:
                break
            # Generate chunks and insert to encoding buffer
            temp_time = next_time
            if next_time % self.seg_duration == self.chunk_duration:
                # If it is the first chunk in a segment
                self.current_seg_idx += 1
                self.current_chunk_idx = 0
                self.generate_chunk_size()
                self.chunks.append([self.current_seg_idx, self.current_chunk_idx, [chunk_size[self.current_chunk_idx] for chunk_size in self.current_seg_size], [np.sum(chunk_size) for chunk_size in self.current_seg_size]])
            else:
                # other chunk in a segment(not first)
                self.current_chunk_idx += 1
                self.chunks.append([self.current_seg_idx, self.current_chunk_idx, [chunk_size[self.current_chunk_idx] for chunk_size in self.current_seg_size]])

    def update(self,downloading_time):
        """
        Description: Update the encoding process
        Args: 
            downloading_time: The current download time(ms)
        Return:
            self.time: The time after downloading chunks
        """
        pre_time = self.time
        self.time += downloading_time
        self.encoding_update(pre_time, self.time)
        return self.time
    
    def get_encoding_buffer_length(self):
        """
        Get the encoding buffer length(current runtime - total segment duration that encoded)
        """
        assert self.chunks[0][1] == 0
        return self.time - self.chunks[0][0] * self.seg_duration
    
    def skip_encoding_buffer(self, target_seg_index):
        """
        Description: Skip some segments to target index
        Args: 
            target_seg_index: download from the target segment index
        Return:
            self.chunks
        """
        while self.chunks and self.chunks[0][0] < target_seg_index:
            self.chunks.pop(0)
    
    def skip(self):
        """
        Description: Skip 2 segments 
        Args: 
            None
        Return:
            time_gap: freeze time
            self.time: the current runtime
        """
        skip_segs = Env_Config.skip_segs
        assert self.chunks[0][1] == 0
        #Change server state to target time after skip
        if self.get_encoding_buffer_length() >= skip_segs * self.seg_duration + self.chunk_duration:
            # If current buffer is long enough
            # There is no freeze. Pop segs from encoding buffer
            target_seg_index = self.chunks[0][0] + skip_segs
            self.skip_encoding_buffer(target_seg_index)
            assert self.chunks[0][0] == target_seg_index and self.chunks[0][1] == 0

            return 0.0, target_seg_index * self.seg_duration
        else:
            # In this case. some existing and future chunks will not be streamed
            time_gap = skip_segs * self.seg_duration + self.chunk_duration - self.get_encoding_buffer_length()
            self.time = (self.chunks[0][0] + skip_segs) * self.seg_duration + self.chunk_duration
            # Purpose of  -1 is to fit encoding_update()
            self.current_seg_idx = self.chunks[0][0] + skip_segs - 1
            # Purpose of set to 4 is to fit encoding_update()
            self.current_chunk_idx = 4
            self.chunks = []
            self.encoding_update(self.time-self.chunk_duration, self.time)

            return time_gap, self.current_seg_idx * self.seg_duration
    
    def generate_next_delivery(self):
        """
        Description: Generate the next delivery info
        Args:
            None
        Return:
            self.next_delivery: The next delivery info [segment_index, chunk_start_index, chunk_end_index, chunks_size(for all bitrates)]
        """
        deliver_chunks = []
        # delivery the first chunk of chunks
        deliver_chunks.append(self.chunks.pop(0))
        # Segment and chunk index of first chunk
        self.next_delivery.extend(deliver_chunks[0][:2])
        # The index of the last chunk
        self.next_delivery.append(deliver_chunks[-1][1])

        # Total size of chunks for each Env_Config.bitrate
        delivery_sizes = []  # chunks size for all bitrates
        for i in range(len(Env_Config.bitrate)):
            delivery_sizes.append(np.sum([chunk[2][i] for chunk in deliver_chunks]))
        self.next_delivery.append(delivery_sizes)
    
    def wait(self):
        """
        Description: Wait for encoding
        Args: 
            None
        Return:
            time_interval: The wait time interval
        """
        next_available_time = (int(self.time/self.chunk_duration) + 1) * self.chunk_duration
        self.encoding_update(self.time, next_available_time)
        assert len(self.chunks) == 1
        time_interval = next_available_time - self.time
        self.time = next_available_time
        return time_interval

    def reset(self, random_latency, testing=False):
        """
        Description: Reset the server
        Args: 
            random_latency: True/False
            testing: Treu/False(default False)
        Return:
            None
        """
        if testing:
            if random_latency:
                self.time = (self.latency_random.randint(Env_Config.server_init_lat_low, Env_Config.server_init_lat_high) + self.latency_random.random()) * Env_Config.seg_duration
            else:
                self.time = (self.latency_random.random() + self.initial_latency) * Env_Config.seg_duration
        else:
            self.time = (self.myRandom.random() + self.initial_latency) * Env_Config.seg_duration
        
        self.current_seg_idx = -1
        self.current_chunk_idx = 0
        self.chunks = []
        self.current_seg_size = [[] for i in range(len(Env_Config.bitrate))]
        self.encoding_update(0.0, self.time)
        del self.next_delivery[:]

    def get_time(self):
        return self.time
    
    def get_next_delivery(self):
        return self.next_delivery
    
    def check_take_action(self):
        """
        Description: Judge the first chunk index of chunks is 0 or not.
        Args:
            None
        Return: 
            True/False
        """
        assert len(self.chunks) >= 1
        if self.chunks[0][1] == 0:
            return True
        else:
            return False
    
    def get_encoding_head_info(self):
        """
        Description: Get the first chunk info(segment index and current chunk index)
        """
        return self.chunks[0][:2]
       
    def check_chunks_empty(self):
        if len(self.chunks) == 0:
            return True
        else:
            return False
    
    def timeout_encoding_buffer(self):
        """
        Description: The timeout of encoding buffer
        Args:
            None
        Return: 
            idx_timeout: the start chunk index for delivery
        """
        temp_seg_index = self.next_delivery[0] # the segment index for delivery
        index_makeup = self.next_delivery[1] # the start chunk index for delivery
        idx_timeout = index_makeup
        while index_makeup >= 0:
            if index_makeup == 0:
                self.chunks.insert(0, [temp_seg_index, index_makeup, [chunk_size[index_makeup] for chunk_size in self.current_seg_size], [np.sum(chunk_size) for chunk_size in self.current_seg_size]])
            else:
                self.chunks.insert(0, [temp_seg_index, index_makeup, [chunk_size[index_makeup] for chunk_size in self.current_seg_size]])
            index_makeup -= 1
        
        return idx_timeout
    
    def clean_next_delivery(self):
        """
        Clean the next delivery list info
        """
        del self.next_delivery[:]