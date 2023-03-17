# -*- encoding: utf-8 -*-
# @Time: 2022/03/24 16:03
# @Author: librah
# @Description: For test task
# @File: test.py
# @Version: 1.0

import numpy as np
from algorithms.lol import LoL_Solver
from algorithms.l2a import L2A_Solver
from algorithms.stallion import Stallion_Solver
from config import Env_Config
from itertools import product


if __name__ == '__main__':

    # seg_size = [[] for i in range(6)]
    # print(len(seg_size[0]))
    # print(int(390/200))
    # a = [5 for _ in range(5)]
    # print(a)
    # temp = []
    # temp.extend([5, 6])
    # print(temp)

    # print(np.round(1.6578, 1))

    # res = lol.LoL_Solver().getProduct(Env_Config.bitrate[:3])
    # print("res: ", res)
    # res2 = list(product(Env_Config.bitrate[:3], repeat=3))
    # print("res2: ", res2)

    # print(float('-inf'))
    # lol_solver = LoL_Solver()
    # quality_index, speed_index = lol_solver.solve(buffer_length=2.5, curr_latency=3.5, player_state=1)

    # l2a_solver = L2A_Solver()
    # quality_index, speed_index = l2a_solver.solve(buffer_length=2.5, curr_latency=3.5, player_state=1)

    stallion_solver = Stallion_Solver()
    quality_index, speed_index = stallion_solver.solve(buffer_length=2.5, curr_latency=3.5)
