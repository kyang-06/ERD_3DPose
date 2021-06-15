import torch
import numpy as np
import time
from tqdm import tqdm
from tool import Bar
import tool.util as util
from dataset.human36m import DataConverter
import pdb

def complete_pelvis_by_z(**kwargs):
    cam_param = kwargs['cam_param']
    pose2d = kwargs['pose2d']
    pelvis_z = kwargs['pelvis_z']

    pelvis_x = (pose2d[:, [0], 0] - cam_param[:, [2]]) * pelvis_z / cam_param[:, [0]]
    pelvis_y = (pose2d[:, [0], 1] - cam_param[:, [3]]) * pelvis_z / cam_param[:, [1]]
    pelvis = torch.cat((pelvis_x, pelvis_y, pelvis_z), -1)
    return pelvis

def fetch_optimized_pelvis(data_loader, opt, selected_jt_ids=(1,4,7)):
    data_converter = DataConverter(opt)
    max_iteration = len(data_loader)
    predict_array = []
    error_proj_array = []
    with torch.no_grad():
        with tqdm(total=max_iteration) as pbar:
            for i, (inps, tars, pel_gt, cam, meta) in enumerate(data_loader):
                batch_size = inps.shape[0]
                inputs = inps.cuda()
                outputs = meta['lifting_result'].cuda()
                outputs[:, 0] = 0
                camera = cam.cuda()

                inputs_unnorm = data_converter.unnormalize(inputs)
                pose_2d = inputs_unnorm
                outputs_unnorm = data_converter.unnormalize(outputs)
                pose_3d = outputs_unnorm

                pelvis, err_proj = util.least_square_pelvis_nonhomo(pose_2d, pose_3d, camera)
                pelvis = pelvis.reshape(batch_size, 1, 3)

                predict_array.append(pelvis.squeeze().cpu().data.numpy())
                error_proj_array.append(err_proj)

                pbar.update(1)

    error_proj_array = np.vstack(error_proj_array)
    print('Error projecting to 2D: %f' % error_proj_array.mean())
    predict_array = np.vstack(predict_array)
    if len(predict_array.shape) == 2:
        predict_array = predict_array[:, np.newaxis, :]     # N*1*3
    return predict_array
