import torch
import numpy as np
import cv2
import pdb
import os

def dict2list(d):
    # convert dict with list items to list with dict items.
    # eg. DL = {'a': [0, 1], 'b': [2, 3]}, LD=[{'a': 0, 'b': 2}, {'a': 1, 'b': 3}]
    # DL to LD
    l = [dict(zip(d,t)) for t in zip(*d.values())]
    return l

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def get_skeleton_orientation(data, parent_idx):
    displacement = data - data[:, parent_idx]
    norm = np.linalg.norm(displacement, axis=-1, keepdims=True) if isinstance(data, np.ndarray) else torch.norm(displacement, dim=-1, keepdim=True)
    norm[norm == 0] = 1
    orientation = displacement / norm
    return orientation

def cam_proj_parallel(global_3d, cam_mat):
    # global_3d: B*J*3
    # cam_mat: B*3*3
    global_3d = global_3d.transpose(2, 1)    # B*3*J
    proj_2d_homo = torch.bmm(cam_mat, global_3d)    # B*3*J
    proj_2d = (proj_2d_homo[:, :2] / proj_2d_homo[:, [2]]).transpose(2, 1)
    return proj_2d

def cam_back_proj_parallel(uv, depth, cam_mat):
    duv = uv * depth
    global_2d_homo = torch.cat((duv, depth), -1).transpose(2,1)
    proj_mat_inv = torch.inverse(cam_mat)
    global_3d = torch.bmm(proj_mat_inv, global_2d_homo).transpose(2,1)
    return global_3d

def get_project_matrix(cam_param, gpu_ver=False):
    # cam_param : B*4
    if isinstance(cam_param, torch.Tensor):
        zero_func = torch.zeros
    elif isinstance(cam_param, np.ndarray):
        zero_func = np.zeros
    else:
        raise Exception('Unknown cam_param type %s' % type(cam_param))

    dim = len(cam_param.shape)
    if dim == 1:
        batch_size = 1
        cam_param = cam_param.reshape(batch_size, -1)
    else:
        batch_size = cam_param.shape[0]

    proj_mat = zero_func([batch_size, 3, 3])
    proj_mat[:, 0, 0] = cam_param[:, 0]  # fx
    proj_mat[:, 1, 1] = cam_param[:, 1]  # fx
    proj_mat[:, 0, 2] = cam_param[:, 2]  # fx
    proj_mat[:, 1, 2] = cam_param[:, 3]  # fx
    proj_mat[:, 2, 2] = 1

    if dim == 1:
        proj_mat = proj_mat.squeeze()

    if gpu_ver:
        proj_mat = proj_mat.cuda()

    return proj_mat

def least_square_pelvis_nonhomo(pose_2d, pose_3d, cam_param, weight=None, selective=False, thres=0.7):
    num_f, num_jts, _ = pose_2d.shape
    err_proj = AverageMeter()
    trans = []
    if isinstance(pose_2d, torch.Tensor):
        pose_2d = pose_2d.data.cpu().numpy()
        pose_3d = pose_3d.data.cpu().numpy()
        cam_param = cam_param.data.cpu().numpy()
    for i in range(num_f):
        if selective:
            sel_ids = np.where(weight[i].reshape(-1) > thres)[0]
            if len(sel_ids) < 6:
                sel_ids = np.argsort(weight[i].reshape(-1))[::-1][:7]
        else:
            sel_ids = range(num_jts)
        cur_num_jts = len(sel_ids)
        cur_pose_2d = pose_2d[i, sel_ids]
        cur_pose_3d = pose_3d[i, sel_ids]
        cam_mat = np.array([[[cam_param[i, 0], 0, cam_param[i, 2]], [0, cam_param[i, 1], cam_param[i, 3]]]])
        A = cam_mat.repeat(cur_num_jts, axis=0)
        A = A.reshape(cur_num_jts*2, 3)
        A[:, -1] -= cur_pose_2d.reshape(-1)
        b = (A[:, -1] * -1) * cur_pose_3d[:, [-1]].repeat(2, axis=-1).reshape(-1) - A[:, :2].sum(-1) * cur_pose_3d[:, :2].reshape(-1)
        b = b.reshape(cur_num_jts*2, 1)
        ATb = np.matmul(A.transpose(), b)       # 3*1
        ATA = np.matmul(A.transpose(), A)       # 3*3
        solution = np.matmul(np.linalg.inv(ATA), ATb)   # 3*1
        trans.append(solution)

        cam_mat = np.concatenate((cam_mat.squeeze(), np.array([[0,0,1]])), 0)
        proj_2d_i = np.matmul(cam_mat, (pose_3d[i].transpose(1, 0) + solution))
        proj_2d_i = proj_2d_i[:2] / proj_2d_i[2:]
        err = np.abs(proj_2d_i - pose_2d[i].transpose(1,0)).mean()
        err_proj.update(err, 1)
    trans = torch.Tensor(trans).cuda()
    return trans, err_proj.avg

def least_square_pelvis_homo(pose_2d, pose_3d, cam_param, weight=None):
    num_f, num_jts, _ = pose_2d.shape
    if weight is None:
        weight = np.ones([num_f, num_jts, 1])
    err_proj = AverageMeter()
    trans = []

    if isinstance(pose_2d, torch.Tensor):
        pose_2d = pose_2d.data.cpu().numpy()
        pose_3d = pose_3d.data.cpu().numpy()
        cam_param = cam_param.data.cpu().numpy()
    for i in range(num_f):
        cam_mat = np.array([[[cam_param[i, 0], 0, cam_param[i, 2]], [0, cam_param[i, 1], cam_param[i, 3]]]])
        A = cam_mat.repeat(num_jts, axis=0)
        A = A.reshape(num_jts * 2, 3)
        A[:, -1] -= pose_2d[i].reshape(-1)
        b = (A[:, -1] * -1) * pose_3d[i, :, [-1]].repeat(2, axis=-1).reshape(-1) - A[:, :2].sum(-1) * pose_3d[i, :,
                                                                                                      :2].reshape(-1)
        b = b.reshape(num_jts * 2, 1)
        Ab = np.concatenate((A, -b), 1)  # (2J)*4
        weighted_Ab = weight[i].repeat(2, axis=0) * Ab

        AbTAb = np.matmul(weighted_Ab.T, weighted_Ab)

        U, s, Vt = np.linalg.svd(AbTAb)
        solution = Vt[-1, :-1] / Vt[-1, -1:]
        solution = solution.reshape(3, 1)
        trans.append(solution)

        cam_mat = np.concatenate((cam_mat.squeeze(), np.array([[0, 0, 1]])), 0)
        proj_2d_i = np.matmul(cam_mat, (pose_3d[i].transpose(1, 0) + solution))
        proj_2d_i = proj_2d_i[:2] / proj_2d_i[2:]
        err = np.abs(proj_2d_i - pose_2d[i].transpose(1, 0)).mean()
        err_proj.update(err, 1)
    trans = torch.Tensor(trans).cuda()
    return trans, err_proj.avg

def least_square_pelvis_parallel(pose_2d, pose_3d, cam_param):
    batch_size, num_jts, _ = pose_2d.shape

    cam_mat = get_project_matrix(cam_param, gpu_ver=True)[:, :2].unsqueeze(1)   # B*1*2*3
    A = cam_mat.repeat([1, num_jts, 1, 1])
    A = A.reshape(batch_size, num_jts*2, 3)
    A[:, :, -1] -= pose_2d.reshape(batch_size, -1)
    b = (A[:, :, -1] * -1) * pose_3d[:, :, [-1]].repeat([1, 1, 2]).reshape(batch_size, -1) - A[:, :, :2].sum(-1) * pose_3d[:, :, :2].reshape(batch_size, -1)
    b = b.reshape(batch_size, num_jts*2, 1)
    ATb = torch.bmm(A.transpose(-1, -2), b)       # 3*1
    ATA = torch.bmm(A.transpose(-1, -2), A)       # 3*3
    trans = torch.bmm(torch.inverse(ATA), ATb)   # 3*1

    return trans

def weighted_least_square_pelvis(pose_2d, pose_3d, cam_param, weight):
    batch_size, num_jts, _ = pose_2d.shape
    line_x = torch.cat((cam_param[:, [0]], torch.zeros_like(cam_param[:, [0]]).cuda(), cam_param[:, [2]]), -1).unsqueeze(1)   # 200x1x3
    line_y = torch.cat((torch.zeros_like(cam_param[:, [1]]).cuda(), cam_param[:, [1]], cam_param[:, [3]]), -1).unsqueeze(1)   # 200x1x3
    A = torch.cat((line_x, line_y), 1).repeat([1, num_jts, 1])
    A = A.reshape(batch_size, num_jts*2, 3)
    A[:, :, -1] -= pose_2d.reshape(batch_size, -1)
    b = (A[:, :, -1] * -1) * pose_3d[:, :, [-1]].repeat([1, 1, 2]).reshape(batch_size, -1) - A[:, :, :2].sum(-1) * pose_3d[:, :, :2].reshape(batch_size, -1)
    b = b.reshape(batch_size, num_jts*2, 1)
    weight = weight.repeat([1, 1, 2]).reshape(batch_size, num_jts*2, 1)       # 200x17x1 -> 200x17x2 -> 200x34x1, for order like (w0, w0, w1, w1, ..., wn, wn)
    A = weight * A
    b = weight * b
    ATb = torch.bmm(A.transpose(-1, -2), b)       # 3*1
    ATA = torch.bmm(A.transpose(-1, -2), A)       # 3*3
    ATA_inv = torch.inverse(ATA)
    solution = torch.bmm(ATA_inv, ATb)   # 3*1
    return solution

def convex_pelvis_z(pose_2d_relative, pose_3d, cam_param):
    num_f, num_jts, _ = pose_2d_relative.shape
    pose_2d_relative = pose_2d_relative.data.cpu().numpy()
    pose_3d = pose_3d.data.cpu().numpy()
    cam_param = cam_param.data.cpu().numpy()
    pelvis_z = []
    for i in range(num_f):
        A = pose_2d_relative[i].reshape(-1)  # (del_u, del_v). 2J
        f = cam_param[i, :2].repeat(num_jts)  # (fx, fy)
        z = pose_3d[i, :, 2].repeat(2)
        xy = pose_3d[i, :, :2].reshape(-1)
        b = f * xy - A * z     # (fx,fy) * (x, y) - (del_u, del_v)*z. 2J
        A = A.reshape(-1, 1)
        b = b.reshape(-1, 1)
        ATb = np.matmul(A.transpose(), b)
        ATA = np.matmul(A.transpose(), A)
        solution = np.matmul(np.linalg.inv(ATA), ATb).squeeze()
        pelvis_z.append(solution.tolist())
    pelvis_z = torch.Tensor(pelvis_z).cuda().reshape(num_f, 1)

    return pelvis_z


def draw_2dpose_on_image(image, pose):
    image = draw_line(image, pose)
    image = draw_point(image, pose)
    return image

def draw_point(image, pose):
    COLOR = (255, 255, 255) #(0, 0, 255) # Red
    # COLOR = (255, 0, 0)
    RADIUS = 8
    for point in pose:
        cv2.circle(image, tuple(point.astype(int)), RADIUS, COLOR, thickness=-1)
    return image

skeleton_st = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8,  9,  8, 11, 12,  8, 14, 15])  # start points
skeleton_ed = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])  # end points
skeleton_lr = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  1,  1], dtype=bool)

#def draw_line(image, pose, lcolor="#3498db", rcolor="#e74c3c"):
def draw_line(image, pose, lcolor=(0, 0, 255), rcolor=(255, 0, 0)):
    THICKNESS = 5
    # color = (255, 0, 0)
    color = (0, 0, 255)

    for i in np.arange(len(skeleton_st)):
        st = tuple(pose[skeleton_st[i]].astype(int).tolist())
        ed = tuple(pose[skeleton_ed[i]].astype(int).tolist())
        #color = lcolor if skeleton_lr[i] else rcolor
        cv2.line(image, st, ed, color, thickness=THICKNESS)
    return image

def show_3dpose(ax, pose, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
    ax.azim = -90
    ax.elev = -70
    # Make connection matrix
    for i in np.arange(len(skeleton_st)):
        x, y, z = [np.array([pose[skeleton_st[i], j], pose[skeleton_ed[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if skeleton_lr[i] else rcolor)

    # add for visualize, should be removed afterward
    ax.plot(pose[[0, 7], 0], pose[[0, 7], 1], pose[[0, 7], 2], c=rcolor)

    # RADIUS = 750 # space around the subject
    RADIUS = np.abs(pose).max() * 1.1
    xroot, yroot, zroot = pose[0, 0], pose[0, 1], pose[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    # Get rid of the ticks and tick labels
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    #
    # ax.get_xaxis().set_ticklabels([])
    # ax.get_yaxis().set_ticklabels([])
    # ax.set_zticklabels([])
    ax.set_aspect('equal')

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)


def pca_transform(model, new_data):
    num_samples = len(new_data)
    new_data = np.vstack(new_data).reshape(num_samples, -1)
    transformed = model.transform(new_data)
    return transformed

def get_bbox(keypoints):
    dim = len(keypoints.shape)
    if isinstance(keypoints, torch.Tensor):
        cat = torch.cat
    else:
        cat = np.concatenate
    if dim == 2:
        keypoints = keypoints.reshape([1] + list(keypoints.shape))
    lt = cat((keypoints[:, :, 0].min(1, keepdims=True), keypoints[:, :, 1].min(1, keepdims=True)), -1)
    rb = cat((keypoints[:, :, 0].max(1, keepdims=True), keypoints[:, :, 1].max(1, keepdims=True)), -1)
    center = (lt + rb) / 2
    scale = rb-lt
    center_scale = cat((center, scale), -1)
    center_scale = center_scale.squeeze()
    return center_scale

def unsqueeze(data, axis):
    if isinstance(data, np.ndarray):
        data = np.expand_dims(data, axis)
    elif isinstance(data, torch.Tensor):
        data = data.unsqueeze(axis)
    return data

def compute_bone_len(pose3d, parent_jt):
    dim = len(pose3d.shape)
    if dim == 2:
        pose3d = unsqueeze(pose3d, 0)
    if isinstance(pose3d, np.ndarray):
        args = {'keepdims':True}
    elif isinstance(pose3d, torch.Tensor):
        args = {'keepdim':True}
    else:
        raise Exception()
    bone_len = ((pose3d - pose3d[:, parent_jt]) ** 2).sum(-1, **args)**0.5
    if dim == 2:
        bone_len = bone_len.squeeze(0)
    return bone_len

def compute_bone_len_bone16(pose3d, parent_jt):
    dim = len(pose3d.shape)
    parent_jt = parent_jt[1:]
    if dim == 2:
        pose3d = unsqueeze(pose3d, 0)
    if isinstance(pose3d, np.ndarray):
        args = {'keepdims':True}
    elif isinstance(pose3d, torch.Tensor):
        args = {'keepdim':True}
    else:
        raise Exception()
    bone_len = ((pose3d[:, 1:] - pose3d[:, parent_jt]) ** 2).sum(-1, **args)**0.5
    if dim == 2:
        bone_len = bone_len.squeeze(0)
    return bone_len

def random_cam_param(batch_size):
    return torch.Tensor([[500, 500, 500, 500]]).cuda().repeat(batch_size, 1).cuda()


def get_procrustes_transformation(X, Y, compute_optimal_scale=False):
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA ** 2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)

    return d, Z, T, b, c