#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
================================================================
Copyright (C) 2019 * Ltd. All rights reserved.
PROJECT      :  RandLA-Net
FILE_NAME    :  inference
AUTHOR       :  DAHAI LU
TIME         :  2020/5/12 下午2:42
PRODUCT_NAME :  PyCharm
================================================================
"""

import os
import time
import numpy as np
from helper_tool import Plot
from helper_ply import read_ply
from helper_tool import DataProcessing as DP
from helper_tool import ConfigSemantic3D as cfg
from DataSet import DataSetSemantic3D
from RandLANet_Inference import Network as NetworkInference
from Semantic3D_Inference import ModelTester as ModelInference


def read_convert_to_array(file):
    pc = read_ply(file)
    pc_xyz = np.vstack((pc['x'], pc['y'], pc['z'])).T
    pc_colors = np.vstack((pc['red'], pc['green'], pc['blue'])).T
    return np.hstack([pc_xyz, pc_colors])


def inference(need_sample, use_votes):
    chosen_snap = MODEL_PATH
    fileList = DP.get_files_list(TEST_PATH, extend)

    #  step 1. raw data load
    t1 = time.time()
    pc_list = []
    for file in fileList:
        if extend == '.ply':
            pc = read_convert_to_array(file)
        elif extend == '.xyz':
            pc = DP.load_pc_semantic3d(file, header=None, delim_whitespace=True)
        else:
            print("unsupported extend : {}".format(extend))
            continue

        pc_list.append(pc)
    t2 = time.time()
    DP.log_string('[raw data load] Done in {:.1f} s\n'.format(t2 - t1), log_out)

    #  step 2. tf data preparation
    t1 = time.time()
    dataset = DataSetSemantic3D(need_sample=need_sample, leaf_size=50)
    dataset.set_input_clouds(pc_list)
    dataset.init_input_pipeline()
    t2 = time.time()
    DP.log_string('[tf data preparation] Done in {:.1f} s\n'.format(t2 - t1), log_out)

    #  step 3. construct network
    t1 = time.time()
    model = NetworkInference(dataset, cfg)
    t2 = time.time()
    DP.log_string('[construct network] Done in {:.1f} s\n'.format(t2 - t1), log_out)

    #  step 4. init network parameters
    t1 = time.time()
    tester = ModelInference(logger=log_out, restore_snap=chosen_snap, on_cpu=False if GPU != -1 else True)
    t2 = time.time()
    DP.log_string('[init network parameters] Done in {:.1f} s\n'.format(t2 - t1), log_out)

    #  step 5. inference
    preds_list = tester.inference(model, dataset, use_votes=use_votes, num_votes=50)

    for file, preds in zip(fileList, preds_list):
        base, _ = os.path.splitext(file)
        label_file_name = base + ".labels"
        np.savetxt(label_file_name, preds, fmt='%d')
        print("Generate labels: {}".format(label_file_name))


def visualization(extend):
    cloud_names = [file_name[:-7] for file_name in os.listdir(TEST_PATH) if file_name[-7:] == '.labels']
    scene_names = []
    label_names = []
    for pc_name in cloud_names:
        if os.path.exists(os.path.join(TEST_PATH, pc_name + extend)):
            scene_names.append(os.path.join(TEST_PATH, pc_name + extend))
            label_names.append(os.path.join(TEST_PATH, pc_name + '.labels'))

    for i in range(len(scene_names)):
        print('scene:', scene_names[i])
        if extend == '.ply':
            data = read_convert_to_array(scene_names[i])
        elif extend == '.xyz':
            data = DP.load_pc_semantic3d(scene_names[i], header=None, delim_whitespace=True)
        else:
            print("unsupported extend : {}".format(extend))
            continue

        pc = data[:, :6].astype(np.float32)
        print('scene point number', pc.shape)
        sem_pred = DP.load_label_semantic3d(label_names[i])
        sem_pred.astype(np.float32)

        # plot
        Plot.draw_pc(pc_xyzrgb=pc[:, 0:6])
        sem_ins_labels = np.unique(sem_pred)
        print('sem_ins_labels: ', sem_ins_labels)
        Plot.draw_pc_sem_ins(pc_xyz=pc[:, 0:3], pc_sem_ins=sem_pred)


if __name__ == '__main__':
    # MODEL_PATH = '/media/yons/data/dataset/pointCloud/RandLA_net/models/Semantic3D/snapshots/snap-29501'
    MODEL_PATH = '/media/yons/data/dataset/pointCloud/RandLA_net/models/Semantic3D/snapshots/snap-42001'
    # MODEL_PATH = '/media/yons/data/dataset/pointCloud/RandLA_net/models/Semantic3D/snapshots/snap-43501' # 0.02
    # MODEL_PATH = '/media/yons/data/dataset/pointCloud/RandLA_net/models/Semantic3D/snapshots/snap-35501' # 0.06
    # MODEL_PATH = '/media/yons/data/dataset/pointCloud/RandLA_net/models/Semantic3D/snapshots/snap-28001'  # 0.04
    DATA_PATH = '/media/yons/data/dataset/pointCloud/data/ownTrainedData'
    TEST_PATH = os.path.join(DATA_PATH, 'test/test_file')
    extend = '.xyz'

    USE_VOTES = False
    NEED_SAMPLE = True

    MODE = 'visualization'  # 'inference' 'visualization'

    GPU = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)

    log_out = open('log_test' + '.txt', 'a')

    if MODE == 'inference':
        start = time.time()
        inference(need_sample=NEED_SAMPLE, use_votes=USE_VOTES)
        end = time.time()
        DP.log_string('total cost time {:.1f} s\n'.format(end - start), log_out)
    elif MODE == 'visualization':
        visualization(extend)
