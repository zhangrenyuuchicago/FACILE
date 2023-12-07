#!/usr/bin/env python

from optparse import OptionParser
import csv
import sys
import os
import tqdm
import glob
import ntpath
import random

import util 
import argparse

def gen_tiles(src_dir, tgt_dir):

    slide_path_lt = glob.glob(src_dir + "/*/*.svs")

    random.shuffle(slide_path_lt)

    def gen_tile_proc(sub_slide_path_lt):
        for file_path in sub_slide_path_lt:
            basename = ntpath.basename(file_path)
            util.gen_tile(file_path, tgt_dir)

    from multiprocessing import Process
    proc_num = 30
    inter = int(len(slide_path_lt) / proc_num) + 1
    p_lt = []

    for i in range(proc_num):
        start = i*inter
        end = (i+1)*inter
        if end > len(slide_path_lt):
            end = len(slide_path_lt)
        sub_slide_path_lt = slide_path_lt[start: end]
        p_lt.append(Process(target=gen_tile_proc, args=(sub_slide_path_lt,)))
        p_lt[i].start()

    for i in range(proc_num):
        p_lt[i].join()

