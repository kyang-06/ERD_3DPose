import os
import torch
import numpy as np
from torch.utils.data import Dataset
import tool.util as util
import pdb
from tqdm import tqdm


S9_drift_fname_list = ['Waiting 1.60457274', 'Greeting.60457274', 'Greeting.58860488', 'SittingDown 1.58860488',
                        'Waiting 1.54138969', 'SittingDown 1.54138969', 'Waiting 1.55011271', 'Greeting.54138969',
                        'Greeting.55011271', 'SittingDown 1.60457274', 'SittingDown 1.55011271', 'Waiting 1.58860488']

jt17_left_idx = [4,5,6,11,12,13]
jt17_right_idx = [1,2,3,14,15,16]

standard_jt17_neighbor_idx = [[0, 1, 4, 7], [1, 0, 2], [2, 1, 3], [3, 2], [4, 0, 5], [5, 4, 6], [6, 5], [7, 0, 8],
                             [8, 7, 9, 11, 14],
                             [9, 8, 10], [10, 9], [11, 8, 12], [12, 11, 13], [13, 12], [14, 8, 15], [15, 14, 16], [16, 15]]

lcn_jt17_adj_idx = [[0, 1, 4, 7], [0, 1, 2, 7], [1, 2, 3], [2, 3],
                    [0, 4, 5, 7], [4, 5, 6], [5, 6], [0, 1, 4, 7, 8, 11, 14],
                    [7, 8, 9, 11, 14], [8, 9, 10], [9, 10], [7, 8, 11, 12],
                    [11, 12, 13], [12, 13], [7, 8, 14, 15], [14, 15, 16], [15, 16]]
jt17_parent_idx = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
#                  jt_group , parent_jt_group
skeleton_graph = [[[1, 4, 7], [0, 0, 0]],
                  [[2, 5, 8], [1, 4, 7]],
                  [[3, 6, 9, 11, 14], [2, 5, 8, 8, 8]],
                  [[10, 12, 15], [9, 11, 14]],
                  [[13, 16], [12, 15]]]


bone_length_by_subject = \
{'S1': np.array([1, 132.9486, 442.8945, 454.2064, 132.9488, 442.8943, 454.2065, 233.4765, 257.0776, 121.1349, 115.0022, 151.0342, 278.8827, 251.7334, 151.0314, 278.8929, 251.7286]),
 'S5': np.array([1, 119.3136, 428.2862, 442.4443, 119.3128, 428.2858, 442.4443, 224.3116, 254.0556, 117.1178, 114.9985, 143.0963, 264.5848, 248.6204, 143.0977, 264.5837, 248.6204]),
 'S6': np.array([1, 142.6138, 486.5709, 461.4936, 142.6117, 486.5637, 461.4939, 262.2149, 260.0092, 119.3989, 115.0, 149.3748, 301.0083, 257.9145, 149.3742, 301.008, 257.9143]),
 'S7': np.array([1, 135.8789, 448.6147, 438.0103, 135.8788, 448.6143, 438.0086, 226.2298, 255.4076, 107.1475, 115.0032, 139.7177, 275.5636, 247.2986, 139.7142, 275.5719, 247.2981]),
 'S8': np.array([1, 146.5378, 452.1476, 438.6334, 146.5366, 452.1463, 438.6333, 261.2141, 251.0243, 120.4586, 115.0003, 169.3163, 289.9085, 244.177, 169.3159, 289.9073, 244.1775]),
 'S9': np.array([1, 124.0746, 472.193, 468.6242, 124.0749, 472.1941, 468.6238, 256.2816, 250.3569, 120.553, 114.9971, 183.2921, 296.3204, 249.0153, 183.2925, 296.3204, 249.0149]),
 'S11': np.array([1, 138.1845, 461.9404, 460.2236, 138.1848, 461.9402, 460.2238, 254.3775, 249.9474, 111.0848, 114.999, 166.3306, 283.1723, 248.3113, 166.3307, 283.1711, 248.3148])}
bone_length_by_subject = {key : util.unsqueeze(val, -1) for key, val in bone_length_by_subject.items()}

class Human36M_OpenPose(Dataset):
    def __init__(self, data_path, is_train, flip=False, exclude_drift_data=False, data_converter=None, opt=None, **kwargs):
        self.data_path = data_path
        self.is_train = is_train
        self.flip = flip

        self.inp, self.out = [], []
        self.bone_len = []
        self.pelvis = [[]]
        self.camera = []
        self.meta = {'info':[]}
        self.confidence_2d = []
        if opt.input_aug or opt.input_aug3:
            self.meta['input_aug'] = []
        if opt.pca_input:
            self.pca_model = kwargs['pca_model']
            self.meta['pca_input'] = []

        self.exclude_drift_data = exclude_drift_data
        self.num_jts = opt.num_jts
        self.out_num_jts = opt.out_num_jts
        self.data_converter = data_converter
        bone_length_by_subject = {}


        # bone_length_fetching = 'fetch_bl' in opt.data_process

        data_prefix = 'train' if is_train else 'test'
        data_2d_file = '%s_custom_2d_unnorm.pth.tar' % data_prefix
        data_3d_file = '%s_custom_3d_unnorm.pth.tar' % data_prefix
        data_2d = torch.load(os.path.join(data_path, data_2d_file))
        data_3d = torch.load(os.path.join(data_path, data_3d_file))

        ordered_key = sorted(data_2d.keys())
        sample_step = opt.train_sample if is_train else opt.test_sample
        counter = 0
        for key in tqdm(ordered_key):
            sub, act, fname = key
            fullact = fname.split('.')[0]
            cam_id = fname.split('.')[-1]   # Direction.xxxxx(hm36) Jog_1.chunk0.1(humaneva)
            if cam_id in util.hm36_cam_table:
                cam_id = util.hm36_cam_table.index(cam_id)
            num_f = data_2d[key].shape[0]
            if (sub == 'S11') and (fullact == 'Directions'):
                continue
            if self.exclude_drift_data and sub == 'S9' and fname in S9_drift_fname_list:
                continue
            for i in range(0, num_f, sample_step):
                p2d_raw = data_2d[key][i]
                if p2d_raw.shape[0] == 25:
                    p2d_raw = op25b_to_hm17(p2d_raw)
                if p2d_raw.shape[1] == 3:
                    p2d_ori = p2d_raw.reshape(self.num_jts, 3)[:, :2]
                    confidence_2d = p2d_raw.reshape(self.num_jts, 3)[:, 2:]
                    self.confidence_2d.append(confidence_2d)
                else:
                    p2d_ori = p2d_raw.reshape(self.num_jts, 2)

                p3d_ori = data_3d[key]['joint_3d'][i].reshape(self.out_num_jts, 3)
                # if i == 0:
                #     print(data_2d[key][i])
                #     import matplotlib.pyplot as plt
                #     plt.figure(0)
                #     plt.scatter(p2d_raw[:, 0], p2d_raw[:, 1]*-1, c='b')
                #     plt.show(block=False)
                #     plt.figure(1)
                #     plt.scatter(data_2d[key][i, :, 0], data_2d[key][i, :, 1]*-1, c='r')
                #     plt.show()
                p2d = self.data_converter.normalize(p2d_ori)
                p3d = self.data_converter.normalize(p3d_ori)
                pelvis = data_3d[key]['pelvis'][i].reshape(1, 3)
                cam_param = np.array([data_3d[key]['camera']['fx'], data_3d[key]['camera']['fy'],
                                      data_3d[key]['camera']['cx'], data_3d[key]['camera']['cy']])
                self.inp.append(p2d)
                self.out.append(p3d)
                self.pelvis[0].append(pelvis)
                self.camera.append(cam_param)
                self.meta['info'].append({'subject':sub, 'action':fullact, 'camid':fname.split('.')[-1], 'frid':i})
                if 'bl' in opt.data_process:
                    bone_len = util.compute_bone_len(p3d_ori, jt17_parent_idx)
                    self.meta['info'][-1]['bone_len'] = bone_len
                if flip:
                    p2d_flip = p2d.copy()  # 1*17*2
                    p3d_flip = p3d.copy()
                    p2d_flip[:, :, 0] *= -1
                    p2d_flip[:, jt17_left_idx + jt17_right_idx] = p2d_flip[:, jt17_right_idx + jt17_left_idx]
                    self.inp.append(p2d_flip)
                    self.out.append(p3d_flip)
                    self.pelvis[0].append(pelvis)
                    self.camera.append(cam_param)
                    self.meta['info'].append({'subject':sub, 'action':fullact, 'camid':fname.split('.')[-1]})
                if 'samplestat2d' in opt.data_process:
                    self.meta['info'][-1]['stat_2d'] = p2d_ori.mean(0).tolist() + p2d_ori.std().tolist()
                if opt.input_aug:
                    p2d_cvt = p2d_ori - cam_param[2:].reshape(1, 2)     # p2d - (cx,cy)
                    mean = np.mean(p2d_cvt, 0)
                    std = np.std(p2d_cvt).reshape(1)
                    # p2d_cvt_normed = (p2d_cvt - mean) / std       # norm to (-1,1)
                    self.meta['input_aug'].append(np.concatenate((p2d_cvt.reshape(-1), mean, std)).astype('float32'))
                if opt.input_aug3:
                    bbox = util.get_bbox(p2d)
                    self.meta['input_aug'].append(np.concatenate((p2d.reshape(-1), bbox.reshape(-1))).astype('float32'))
                if opt.pca_input:
                    pca_feat = util.pca_transform(self.pca_model, p2d.reshape(1, -1)).squeeze()
                    self.meta['pca_input'].append(pca_feat.astype('float32'))
            counter += 1        # to prevent error when conducting if expression
            if opt.debug_speedup and counter > 0:
                break


    def __getitem__(self, index):
        inputs = torch.Tensor(self.inp[index]).float()
        outputs = torch.Tensor(self.out[index]).float()
        pelvis = torch.Tensor([pi[index] for pi in self.pelvis]).float()
        camera = torch.Tensor(self.camera[index])
        meta = {key: self.meta[key][index] for key in self.meta}
        if len(self.confidence_2d) > 0:
            meta['confidence_2d'] = self.confidence_2d[index]
        return inputs, outputs, pelvis, camera, meta

    def __len__(self):
        return len(self.inp)

def is_valid_keypoint(keypoint):
    if (np.abs(keypoint) < 1e-2).all():
        return False
    else:
        return True

def op25_to_hm17(op25):
    # hm 17 jt: 'Pelvis', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'torso', 'neck', 'nose', 'head', 0-10
    #           'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist' 11-16
    # op 25 jt: 'nose', 'neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 0-7
    #           'pelvis', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar', 8-18
    #           'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel' 19-24
    dim = op25.shape[-1]
    hm17 = np.zeros([17, dim])
    hm17[0] = op25[8]   # pelvis
    hm17[1:4] = op25[9:12]  # RLeg
    hm17[4:7] = op25[12:15] # LLeg
    hm17[7, :2] = op25[[1, 8], :2].mean(0)    # torso: (neck+pelvis)/2
    hm17[8] = op25[1]   # neck
    hm17[9] = op25[0]   # nose
    hm17[10, :2] = op25[0, :2] - (op25[15:17, :2].mean(0) - op25[0, :2]) * 2 # head: (eye_center-nose)*2 + nose
    hm17[11:14] = op25[5:8]     # LArm
    hm17[14:17] = op25[2:5]     # RArm
    if dim == 3:
        hm17[10, 2] = 0
        hm17[7, 2] = 0
    return hm17

def op25b_to_hm17(op25b):
    # op 25b jt:    'nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist, 0-10
    #               'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle', 'neck', 'head', 11-18
    #               'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel' 19-24
    dim = op25b.shape[-1]
    hm17 = np.zeros([17, dim])
    hm17[0] = op25b[11:13].mean(0) if is_valid_keypoint(op25b[11]) and is_valid_keypoint(op25b[12]) else 0  # pelvis
    hm17[1:4] = op25b[[12, 14, 16]]  # RLeg
    hm17[4:7] = op25b[[11, 13, 15]]  # LLeg
    hm17[7] = (op25b[17] + hm17[0]) / 2 if is_valid_keypoint(op25b[11]) and is_valid_keypoint(op25b[12]) and is_valid_keypoint(op25b[17]) else 0 # torso: (neck+pelvis)/2
    hm17[8] = op25b[17]  # neck
    hm17[9] = op25b[0] # nose
    hm17[10] = op25b[18]    # head
    hm17[11:14] = op25b[[5, 7, 9]]  # LArm
    hm17[14:17] = op25b[[6, 8, 10]]  # RArm
    # if dim == 3:
    #     hm17[0, 2] = 0
    #     hm17[7, 2] = 0
    return hm17
