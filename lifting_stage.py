import time
import torch
from tool import Bar
import numpy as np
import tool.util as util
from network.linear_model import LinearModel
from network.gcn_model import SemGCN
from network.lcn_model import LCNModel
from collections import OrderedDict
from dataset.human36m import DataConverter, standard_jt17_neighbor_idx
from base_modules import get_knn_mat
import copy
import pdb

def get_lifting_model(opt):
    print(">>> Creating lifting model...")
    num_jts = opt.num_jts
    out_num_jts = opt.out_num_jts
    input_dim = 2
    output_dim = 3
    if opt.pca_input:
        num_jts = 1
        out_num_jts = 1
        input_dim = int(opt.pca_component)
        output_dim = opt.out_num_jts * 3

    if opt.lifting_model == 'linear':
        SingleModel = LinearModel
        args = dict(input_dim=input_dim, output_dim=output_dim, hidden_size=opt.hidsize, num_block=opt.num_block,
                    num_jts=num_jts, out_num_jts=out_num_jts, p_dropout=opt.dropout)
    elif opt.lifting_model == 'lcn':
        SingleModel = LCNModel
        args = dict(input_dim=input_dim, output_dim=output_dim, hidden_size=opt.hidsize, num_block=opt.num_block,
                    num_jts=num_jts, p_dropout=opt.dropout, knn=opt.knn)
    elif opt.lifting_model == 'semgcn':
        SingleModel = SemGCN
        adj = torch.Tensor(get_knn_mat(1, adj_idx=standard_jt17_neighbor_idx))
        args = dict(adj=adj, hid_dim = int(opt.hidsize), coords_dim = (input_dim, 3), num_layers = opt.num_block,
                    non_local = True)
    else:
        raise Exception('Unknown lifting network %s' % opt.lifting_model)

    lifting_model = SingleModel(**args)
    if opt.gpu != '-1':
        lifting_model = lifting_model.cuda()
    params_num = sum([p.numel() for p in lifting_model.parameters()]) / 1.0e6
    print(">>> lifting model params: {:.2f}M".format(params_num))

    if opt.load:
        if opt.use_gpu:
            ckpt = torch.load(opt.load)
        else:
            ckpt = torch.load(opt.load, map_location='cpu')
        print(">>> lifting model ckpt loaded (error: {})".format(ckpt['error']))
        state_dict = ckpt['state_dict']
        lifting_model.load_state_dict(state_dict)

    return lifting_model

def train_lifting(epoch, train_loader, lifting_model, criterion, optimizer, opt):
    losses = util.AverageMeter()

    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(train_loader))

    lifting_model.train()

    for i, (inputs, targets, _, _, meta) in enumerate(train_loader):
        batch_size = targets.shape[0]
        inputs_gpu = inputs.cuda() if opt.use_gpu else inputs
        targets_gpu = targets.cuda() if opt.use_gpu else targets
        if opt.pca_input:
            pca_input = meta['pca_input'].cuda() if opt.use_gpu else meta['pca_input']
            inputs_gpu = pca_input

        outputs_gpu = lifting_model(inputs_gpu)
        optimizer.zero_grad()
        loss = criterion(outputs_gpu, targets_gpu)
        losses.update(loss.item(), batch_size)
        loss.backward()

        if opt.max_norm:
            lifting_model.grad_clip()
        optimizer.step()

        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.1}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.6f}' \
            .format(batch=i + 1,
                    size=len(train_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()
    bar.finish()
    return losses.avg


def test_lifting(epoch, test_loader, lifting_model, criterion, opt):
    loss, outputs_array, targets_array, corresponding_info = inference_lifting(test_loader, lifting_model, opt, criterion)
    num_sample = len(outputs_array)
    data_converter = DataConverter(opt)

    outputs_array_by_action = rearrange_by_key(outputs_array, corresponding_info)
    targets_array_by_action = rearrange_by_key(targets_array, corresponding_info)
    corresponding_info_by_action = rearrange_by_key(corresponding_info, copy.deepcopy(corresponding_info))


    if opt.eval_jt:
        evaluate_jtwise(outputs_array, targets_array, opt)

    # evaluate
    err_ttl, err_act, err_dim = evaluate_actionwise(outputs_array_by_action, targets_array_by_action, opt,
                                                    corresponding_info=corresponding_info_by_action)

    print(">>> error mean of %d samples: %.3f <<<" % (num_sample, err_ttl))
    print(">>> error by dim: x: %.3f,  y:%.3f, z:%.3f <<<" % (tuple(err_dim)))

    if opt.test and opt.stage == 'lifting':
        if 'humaneva' in opt.data_rootdir:
            evaluate_humaneva(outputs_array_by_action, targets_array_by_action, opt, corresponding_info=corresponding_info_by_action, need_unnormalize=True)
        else:
            print(">>> error by action")
            for key in sorted(err_act.keys()):
                val = err_act[key]
                print("%s: %.2f" % (key, val))

    return loss, err_ttl, err_dim

def rearrange_by_key(array, guide_info, guide_key='action'):
    arr_rearrange = {}
    for i in range(len(array)):
        key = guide_info[i][guide_key]
        if key not in arr_rearrange:
            arr_rearrange[key] = []
        arr_rearrange[key].append(array[i])
    for key in arr_rearrange:
        arr_rearrange[key] = np.array(arr_rearrange[key])
    return arr_rearrange

def inference_lifting(test_loader, lifting_model, opt, criterion=None):
    losses = util.AverageMeter()

    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(test_loader))

    # inference
    lifting_model.eval()
    outputs_array = []
    targets_array = []
    corresponding_info = []
    with torch.no_grad():
        for i, (inputs, targets, _, _, meta) in enumerate(test_loader):
            batch_size = targets.shape[0]
            inputs_gpu = inputs.cuda() if opt.use_gpu else inputs
            targets_gpu = targets.cuda() if opt.use_gpu else targets
            if opt.pca_input:
                pca_input = meta['pca_input'].cuda() if opt.use_gpu else meta['pca_input']
                inputs_gpu = pca_input
            info = meta['info']
            info['fullaction'] = info['action']
            info['action'] = list(map(lambda x: x.split(' ')[0], info['action']))
            outputs_gpu = lifting_model(inputs_gpu)
            if criterion is not None:
                loss = criterion(outputs_gpu, targets_gpu)
                losses.update(loss.item(), batch_size)

            outputs_array.append(outputs_gpu.cpu().data.numpy())
            targets_array.append(targets.data.numpy())
            info_list = util.dict2list(info)
            corresponding_info += info_list

            del inputs_gpu, targets_gpu, info

            # update summary
            if (i + 1) % 100 == 0:
                batch_time = time.time() - start
                start = time.time()

            bar.suffix = '({batch}/{size}) | batch: {batchtime:.1}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.6f}' \
                .format(batch=i + 1,
                        size=len(test_loader),
                        batchtime=batch_time * 10.0,
                        ttl=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg)
            bar.next()
    bar.finish()

    outputs_array = np.vstack(outputs_array)
    targets_array = np.vstack(targets_array)
    return losses.avg, outputs_array, targets_array, corresponding_info


def evaluate_actionwise(outputs_array, targets_array, opt, **kwargs):
    data_converter = DataConverter(opt)
    err_ttl = util.AverageMeter()
    err_act = {}
    err_dim = [util.AverageMeter(), util.AverageMeter(), util.AverageMeter()]

    need_unnormalize = True if 'need_unnormalize' not in kwargs or kwargs['need_unnormalize'] else False
    for act in sorted(outputs_array.keys()):
        num_sample = outputs_array[act].shape[0]
        predict = data_converter.unnormalize(outputs_array[act]) if need_unnormalize else outputs_array[act]
        # gt should not be unnormalized with avg bone len
        gt = data_converter.unnormalize(targets_array[act]) if need_unnormalize else targets_array[act]

        if opt.procrustes:
            pred_procrustes = []
            for i in range(num_sample):
                _, Z, T, b, c = util.get_procrustes_transformation(gt[i], predict[i], True)
                pred_procrustes.append((b * predict[i].dot(T)) + c)
            predict = np.array(pred_procrustes)

        err_act[act] = (((predict - gt) ** 2).sum(-1)**0.5).mean()
        err_ttl.update(err_act[act], 1)
        for dim_i in range(len(err_dim)):
            err = (np.abs(predict[:, :, dim_i] - gt[:, :, dim_i])).mean()
            err_dim[dim_i].update(err, 1)

    for dim_i in range(len(err_dim)):
        err_dim[dim_i] = err_dim[dim_i].avg

    return err_ttl.avg, err_act, err_dim

def evaluate_humaneva(outputs_array, targets_array, opt, **kwargs):
    data_converter = DataConverter(opt)
    err_ttl = util.AverageMeter()
    err_dim = [util.AverageMeter(), util.AverageMeter(), util.AverageMeter()]
    err_table = {}

    corresponding_info = kwargs['corresponding_info'] if 'corresponding_info' in kwargs else None
    need_unnormalize = True if 'need_unnormalize' not in kwargs or kwargs['need_unnormalize'] else False
    for act in sorted(outputs_array.keys()):
        if act not in err_table:
            err_table[act] = {}
        for sample_i in range(len(outputs_array[act])):
            subject = corresponding_info[act][sample_i]['subject']
            if subject not in err_table[act]:
                err_table[act][subject] = util.AverageMeter()
            predict = data_converter.unnormalize(outputs_array[act][sample_i]) if need_unnormalize else outputs_array[act][sample_i]
            # gt should not be unnormalized with avg bone len
            gt = data_converter.unnormalize(targets_array[act][sample_i]) if need_unnormalize else targets_array[act][sample_i]

            if opt.procrustes:
                pred_procrustes = []
                _, Z, T, b, c = util.get_procrustes_transformation(gt, predict, True)
                pred_procrustes.append((b * predict.dot(T)) + c)
                predict = pred_procrustes
            error = (((predict - gt) ** 2).sum(-1)**0.5).mean()
            err_table[act][subject].update(error, 1)
            err_ttl.update(error, 1)
    for act in sorted(err_table.keys()):
        for subject in sorted(err_table[act].keys()):
            print('%s - %s: %.3f' % (act, subject, err_table[act][subject].avg))

    return err_ttl.avg, err_table, None



def evaluate_jtwise(outputs_array, targets_array, opt):
    idx2name = ['root', 'R-hip', 'R-knee', 'R-ankle', 'L-hip', 'L-knee', 'L-ankle', 'torso', 'neck', 'nose',
                'head', 'L-shoulder', 'L-elbow', 'L-wrist', 'R-shoulder', 'R-elbow', 'R-wrist', 'thorax']
    data_converter = DataConverter(opt)
    outputs_array = data_converter.unnormalize(outputs_array)
    targets_array = data_converter.unnormalize(targets_array)
    err_jt = (((outputs_array - targets_array) ** 2).sum(-1)**0.5).mean(0)      # J
    str_err_jt = map(lambda x: '%.3f' % x, err_jt)
    print('>>> error by joint')
    print(OrderedDict(zip(idx2name, str_err_jt)))