import time
import torch
import numpy as np
import tool.util as util
from tool import Bar

from lifting_stage import inference_lifting, rearrange_by_key, evaluate_actionwise, evaluate_humaneva
from network.residual_regressor_model import ResidualRegressor
from pelvis_modules import fetch_optimized_pelvis
from dataset.human36m import DataConverter,standard_jt17_neighbor_idx
import copy
import os

def get_increment_model(opt):
    print(">>> Creating increment model...")
    num_jts = opt.num_jts
    output_dim = 3
    if opt.inc_input_type in ['delta', 'ori', 'proj']:
        if opt.pca_input:
            input_dim = int(opt.pca_component)
            output_dim = opt.num_jts * 3
            num_jts = 1
        else:
            input_dim = 2
    elif opt.inc_input_type in ['cat', 'catproj']:
        if not opt.pca_input:
            input_dim = 4
        else:
            input_dim = int(opt.pca_component) * 2
            output_dim = opt.num_jts * 3
            num_jts = 1
    else:
        raise Exception('Unknown inc_input_type %s' % opt.inc_input_type)
    increment_model = ResidualRegressor(input_dim=input_dim, output_dim=output_dim, hidden_size=opt.inc_hidsize, num_block=opt.inc_block, num_jts=num_jts,
                                        inc_num=opt.inc_num,
                                        p_dropout=opt.dropout)
    if opt.use_gpu:
        increment_model = increment_model.cuda()
    params_num = sum([p.numel() for p in increment_model.parameters()]) / 1.0e6
    print(">>> increment model params: {:.2f}M".format(params_num))

    if opt.load_inc:
        if opt.use_gpu:
            ckpt = torch.load(opt.load_inc)
        else:
            ckpt = torch.load(opt.load_inc, map_location='cpu')
        print(">>> increment ckpt loaded (error: {})".format(ckpt['error']))
        state_dict = ckpt['state_dict']
        increment_model.load_state_dict(state_dict)
        inc_num_best = ckpt['inc_num']
        if opt.test:
            # drop redundant increment module
            increment_model.model = increment_model.model[:inc_num_best]
    return increment_model


def prepare_increment(data_loader, lifting_model, opt):
    # infer lifting model result
    dataset = data_loader.dataset
    print(">>> Infering lifting model for initial result...")
    _, outputs, _, _ = inference_lifting(data_loader, lifting_model, opt)
    print("Storing predicts of %d samples..." % len(outputs))
    dataset.meta['lifting_result'] = outputs

    update_pelvis_position(data_loader, opt)


def update_pelvis_position(data_loader, opt, predict_pose3d=None):
    dataset = data_loader.dataset
    if predict_pose3d is not None:
        # trick way to update pelvis with predict pose3d via replacing 'lifting_model_result' by predict_pose3d
        lifting_result = dataset.meta['lifting_result'].copy()
        dataset.meta['lifting_result'] = predict_pose3d

    if opt.optim_pelvis:
        print(">>> Replacing groudth truth pelvis with optimized predict...")
        gt_pelvis = np.array(dataset.pelvis[0]).copy()
        selected_ids = range(0, opt.num_jts)
        pred_pelvis = fetch_optimized_pelvis(data_loader, opt, selected_ids)

        err_pel = np.abs(gt_pelvis - pred_pelvis).squeeze().mean(0)
        print("mean error of predict pelvis with ids %s: " % str(selected_ids))
        print("x: %.3f, y: %.3f, z: %.3f" % tuple(err_pel.tolist()))
        dataset.pelvis.append(pred_pelvis)

    if predict_pose3d is not None:
        dataset.meta['lifting_result'] = lifting_result


def train_increment(epoch, train_loader, increment_list, criterion, optimizer, opt, **kwargs):
    losses = {'xy': util.AverageMeter(), 'z': util.AverageMeter()}
    inc_error = util.AverageMeter()
    lift_error = util.AverageMeter()
    inc_nth = len(increment_list)
    stack_args = {}
    for i in range(inc_nth-1):
        increment_list[i].eval()
    increment_list[inc_nth - 1].train()

    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(train_loader))

    if opt.pca_input:
        stack_args['pca_model'] = train_loader.dataset.pca_model

    data_converter = DataConverter(opt)

    for i, (inputs, targets, pelvis, cam, meta) in enumerate(train_loader):
        batch_size = targets.shape[0]
        inputs_gpu = inputs.cuda() if opt.use_gpu else inputs
        targets_gpu = targets.cuda() if opt.use_gpu else targets
        cam_param = cam.cuda() if opt.use_gpu else cam
        cam_mat = util.get_project_matrix(cam_param, gpu_ver=opt.use_gpu)
        outputs_lifting_gpu = meta['lifting_result'].cuda() if opt.use_gpu else meta['lifting_result']

        pelvis_gpu = pelvis.cuda()

        if opt.pca_input:
            pca_input = meta['pca_input'].cuda() if opt.use_gpu else meta['pca_input']
            stack_args['pca_input'] = pca_input
            inputs_gpu = meta['pca_input'].cuda() if opt.use_gpu else meta['pca_input']

        pred_lifting = data_converter.unnormalize(outputs_lifting_gpu)
        gt = data_converter.unnormalize(targets_gpu)
        inc_outputs = None
        accu_pred = pred_lifting.clone()

        for j in range(inc_nth):
            if opt.optim_pelvis:
                if opt.update_pelvis:
                    cur_pelvis = pelvis_gpu[:, j+1]
                else:
                    cur_pelvis = pelvis_gpu[:, 1]
            else:
                cur_pelvis = pelvis_gpu[:, 0]
            args = dict(model=increment_list[j], inputs=inputs_gpu, accu_predict=accu_pred,
                        pelvis=cur_pelvis, cam_mat=cam_mat, data_converter=data_converter, opt=opt, **stack_args)
            if j < inc_nth - 1:
                with torch.no_grad():
                    _, accu_pred = stack_result(**args)
            else:
                _, accu_pred = stack_result(**args)
        inc_targets = gt - accu_pred

        optimizer.zero_grad()
        loss = criterion(accu_pred, gt)
        loss_xy = criterion(accu_pred[:, :, :2], gt[:, :, :2])
        loss_z = criterion(accu_pred[:, :, 2:], gt[:, :, 2:])
        losses['xy'].update(loss_xy, batch_size)
        losses['z'].update(loss_z, batch_size)
        loss.backward()
        if opt.max_norm:
            torch.nn.utils.clip_grad_norm_(increment_list[-1].parameters(), max_norm=1)
        optimizer.step()

        # calculate accumulated iteration error
        cur_inc_err = (((accu_pred - gt) ** 2).sum(-1)**0.5).mean()
        cur_lift_err = (((pred_lifting - gt) ** 2).sum(-1) ** 0.5).mean()
        inc_error.update(cur_inc_err, batch_size)
        lift_error.update(cur_lift_err, batch_size)

        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

        del inputs_gpu, targets_gpu, cam_param, outputs_lifting_gpu, cam_mat, accu_pred, meta, pelvis_gpu
        if opt.pca_input:
            del pca_input
        torch.cuda.empty_cache()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.1f}ms | Total: {ttl} | ETA: {eta:} | loss: {loss_xy:.4f}, {loss_z:.4f} ' \
                     '| inc_error: {ierror:.2f} | lift_error: {lerror:.2f} ' \
            .format(batch=i + 1,
                    size=len(train_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss_xy=losses['xy'].avg,
                    loss_z=losses['z'].avg,
                    ierror=inc_error.avg,
                    lerror=lift_error.avg)
        bar.next()
    bar.finish()

    return losses['xy'].avg + losses['z'].avg

def test_increment(epoch, test_loader, increment_list, criterion, opt, **kwargs):
    loss, outputs_array, targets_array, corresponding_info = inference_increment(test_loader, increment_list,
                                                                                 criterion, opt, **kwargs)
    outputs_array_by_action = rearrange_by_key(outputs_array, corresponding_info)
    targets_array_by_action = rearrange_by_key(targets_array, corresponding_info)
    corresponding_info_by_action = rearrange_by_key(corresponding_info, copy.deepcopy(corresponding_info))

    # evaluate
    err_ttl, err_act, err_dim = evaluate_actionwise(outputs_array_by_action, targets_array_by_action, opt,
                                                    corresponding_info=corresponding_info_by_action, need_unnormalize=True)

    print(">>> error mean: %.3f <<<" % err_ttl)
    print(">>> error by dim: x: %.3f,  y:%.3f, z:%.3f <<<" % (tuple(err_dim)))

    if opt.test:
        if 'humaneva' in opt.data_rootdir:
            evaluate_humaneva(outputs_array_by_action, targets_array_by_action, opt, corresponding_info=corresponding_info_by_action, need_unnormalize=False)
        else:
            print(">>> error by action")
            for key in sorted(err_act.keys()):
                val = err_act[key]
                print("%s: %.2f" % (key, val))

    return loss, err_ttl, err_dim

def inference_increment(test_loader, increment_list, criterion, opt, **kwargs):
    losses = {'xy': util.AverageMeter(), 'z': util.AverageMeter()}
    inc_error = util.AverageMeter()
    lift_error = util.AverageMeter()
    inc_nth = len(increment_list)
    stack_args = {}
    for i in range(inc_nth):
        increment_list[i].eval()

    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(test_loader))

    if opt.pca_input:
        stack_args['pca_model'] = test_loader.dataset.pca_model

    data_converter = DataConverter(opt)
    predicts_array = []
    targets_array = []
    corresponding_info = []
    with torch.no_grad():
        for i, (inputs, targets, pelvis, cam, meta) in enumerate(test_loader):
            batch_size = targets.shape[0]
            inputs_gpu = inputs.cuda() if opt.use_gpu else inputs
            targets_gpu = targets.cuda() if opt.use_gpu else targets
            cam_param = cam.cuda() if opt.use_gpu else cam
            outputs_lifting_gpu = meta['lifting_result'].cuda() if opt.use_gpu else meta['lifting_result']
            cam_mat = util.get_project_matrix(cam_param, gpu_ver=opt.use_gpu)
            info = meta['info']
            info['fullaction'] = info['action']
            info['action'] = list(map(lambda x: x.split(' ')[0], info['action']))
            pelvis_gpu = pelvis.cuda() if opt.use_gpu else pelvis
            if opt.pca_input:
                stack_args['pca_input'] = meta['pca_input'].cuda() if opt.use_gpu else meta['pca_input']
                inputs_gpu = meta['pca_input'].cuda() if opt.use_gpu else meta['pca_input']

            pred_lift = data_converter.unnormalize(outputs_lifting_gpu)
            gt = data_converter.unnormalize(targets_gpu)
            inc_outputs = None
            accu_pred = pred_lift.clone()

            for j in range(inc_nth):
                if opt.optim_pelvis:
                    if opt.update_pelvis:
                        cur_pelvis = pelvis_gpu[:, j+1]
                    else:
                        cur_pelvis = pelvis_gpu[:, 1]
                else:
                    cur_pelvis = pelvis_gpu[:, 0]
                args = dict(model=increment_list[j], inputs=inputs_gpu, accu_predict=accu_pred,
                            pelvis=cur_pelvis, cam_mat=cam_mat, data_converter=data_converter, opt=opt, **stack_args)
                _, accu_pred = stack_result(**args)

            loss = criterion(accu_pred, gt)
            loss_xy = criterion(accu_pred[:, :, :2], gt[:, :, :2])
            loss_z = criterion(accu_pred[:, :, 2:], gt[:, :, 2:])
            losses['xy'].update(loss_xy, batch_size)
            losses['z'].update(loss_z, batch_size)


            # calculate accumulated iteration error
            cur_inc_err = (((accu_pred - gt) ** 2).sum(-1)**0.5).mean()
            cur_lift_err = (((pred_lift - gt) ** 2).sum(-1) ** 0.5).mean()
            inc_error.update(cur_inc_err, batch_size)
            lift_error.update(cur_lift_err, batch_size)

            # save normalized predicts
            predicts_array.append(data_converter.normalize(accu_pred).cpu().data.numpy())
            targets_array.append(data_converter.normalize(gt).cpu().data.numpy())
            info_list = util.dict2list(info)
            corresponding_info += info_list

            del inputs_gpu, targets_gpu, cam_param, cam_mat, pelvis_gpu, info
            torch.cuda.empty_cache()

            # update summary
            if (i + 1) % 100 == 0:
                batch_time = time.time() - start
                start = time.time()
            bar.suffix = '({batch}/{size}) | batch: {batchtime:.1f}ms | Total: {ttl} | ETA: {eta:} | loss: {loss_xy:.4f}, {loss_z:.4f} ' \
                         '| inc_error: {ierror:.2f} | lift_error: {lerror:.2f} ' \
                .format(batch=i + 1,
                        size=len(test_loader),
                        batchtime=batch_time * 10.0,
                        ttl=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss_xy=losses['xy'].avg,
                        loss_z=losses['z'].avg,
                        ierror=inc_error.avg,
                        lerror=lift_error.avg)
            bar.next()
    bar.finish()
    predicts_array = np.vstack(predicts_array)
    targets_array = np.vstack(targets_array)
    return losses['xy'].avg + losses['z'].avg, predicts_array, targets_array, corresponding_info


def stack_result(model, inputs, accu_predict, pelvis, cam_mat, data_converter, opt, **kwargs):
    # 3d_0 projection to 2d_1
    # delta_2d <- 2d_0 - 2d_1
    # delta_3d <- 3d_gt - 3d_0
    batch_size = inputs.shape[0]
    global_3d = accu_predict + pelvis
    if not opt.pca_input:
        proj_2d = util.cam_proj_parallel(global_3d, cam_mat)
        proj_2d_normed = data_converter.normalize(proj_2d)
        delta = inputs - proj_2d_normed
        if opt.inc_input_type == 'delta':
            inc_inputs = delta
        elif opt.inc_input_type == 'cat':
            inc_inputs = torch.cat((inputs, delta), -1)
        elif opt.inc_input_type == 'catproj':
            inc_inputs = torch.cat((inputs, proj_2d_normed), -1)
        elif opt.inc_input_type == 'ori':
            inc_inputs = inputs
        elif opt.inc_input_type == 'proj':
            inc_inputs = proj_2d_normed
        else:
            raise Exception('Unknown opt.inc_input_type %s' % opt.inc_input_type)
    else:
        pca_model = kwargs['pca_model']
        proj_2d = util.cam_proj_parallel(global_3d, cam_mat)
        proj_2d_normed = data_converter.normalize(proj_2d)
        proj_2d_pca = torch.mm((proj_2d_normed.reshape(batch_size, -1))-pca_model.mean_gpu, pca_model.components_gpu.transpose(1,0))
        delta = inputs - proj_2d_pca
        if opt.inc_input_type == 'delta':
            inc_inputs = delta     # inputs == inputs_pca
        elif opt.inc_input_type == 'cat':
            inc_inputs = torch.cat((inputs, delta), -1)
        elif opt.inc_input_type == 'catproj':
            inc_inputs = torch.cat((inputs, proj_2d_pca), -1)
        elif opt.inc_input_type == 'ori':
            inc_inputs = inputs
        elif opt.inc_input_type == 'proj':
            inc_inputs = proj_2d_pca
        else:
            raise Exception('Unknown opt.inc_input_type %s' % opt.inc_input_type)
        del proj_2d_pca
    inc_outputs = model(inc_inputs)
    accu_predict += inc_outputs

    del proj_2d, proj_2d_normed, global_3d
    torch.cuda.empty_cache()

    return inc_outputs, accu_predict


def inference_proj(test_loader, opt):
    proj_error = util.AverageMeter()
    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(test_loader))

    data_converter = DataConverter(opt)
    predicts_array = []
    targets_array = []
    corresponding_action = []

    with torch.no_grad():
        for i, (inputs, targets, pelvis, cam, meta) in enumerate(test_loader):
            batch_size = targets.shape[0]
            inputs_gpu = inputs.cuda() if opt.use_gpu else inputs
            targets_gpu = targets.cuda() if opt.use_gpu else targets
            pelvis_gpu = pelvis.cuda() if opt.use_gpu else pelvis
            outputs_lifting_gpu = meta['lifting_result'].cuda() if opt.use_gpu else meta['lifting_result']
            cam_mat = util.get_project_matrix(cam, gpu_ver=opt.use_gpu)
            info = meta['info']
            action = list(map(lambda x: x.split(' ')[0], info['action']))

            # generate predict Z
            pred_lift = data_converter.unnormalize(outputs_lifting_gpu)
            inputs_unnorm_gpu = data_converter.unnormalize(inputs_gpu)
            predict_depth = (pred_lift + pelvis_gpu)[:, :, 2:]
            # (u,v,d) -> (x,y,z)
            predict_abs = util.cam_back_proj_parallel(inputs_unnorm_gpu, predict_depth, cam_mat)

            targets_unnorm_gpu = data_converter.unnormalize(targets_gpu)
            gt_abs = targets_unnorm_gpu + pelvis_gpu
            err = (((predict_abs - gt_abs) ** 2).sum(-1) ** 0.5).mean()

            # calculate accumulated iteration error
            proj_error.update(err, batch_size)

            predicts_array.append(predict_abs.data.cpu().numpy())
            targets_array.append(gt_abs.data.cpu().numpy())
            corresponding_action += action

            # update summary
            if (i + 1) % 100 == 0:
                batch_time = time.time() - start
                start = time.time()
            bar.suffix = '({batch}/{size}) | batch: {batchtime:.1f}ms | Total: {ttl} | ETA: {eta:} | error: {error:.3f}' \
                .format(batch=i + 1,
                        size=len(test_loader),
                        batchtime=batch_time * 10.0,
                        ttl=bar.elapsed_td,
                        eta=bar.eta_td,
                        error=proj_error.avg)
            bar.next()
    bar.finish()
    predicts_array = np.vstack(predicts_array)
    targets_array = np.vstack(targets_array)
    return predicts_array, targets_array, corresponding_action

def test_proj(test_loader, opt):
    predicts_array, targets_array, corresponding_action = inference_proj(test_loader, opt)
    # pose-wise array to action-wise dict
    predicts_array_by_action = {}
    targets_array_by_action = {}
    for i in range(len(predicts_array)):
        act = corresponding_action[i]
        if act not in predicts_array_by_action:
            predicts_array_by_action[act] = []
            targets_array_by_action[act] = []
        predicts_array_by_action[act].append(predicts_array[i])
        targets_array_by_action[act].append(targets_array[i])
    for act in predicts_array_by_action:
        predicts_array_by_action[act] = np.array(predicts_array_by_action[act])
        targets_array_by_action[act] = np.array(targets_array_by_action[act])

    err_ttl, err_act, err_dim = evaluate_actionwise(predicts_array_by_action, targets_array_by_action)
    print(">>> error mean: %.3f <<<" % err_ttl)
    print(">>> error by dim: x: %.3f,  y:%.3f, z:%.3f <<<" % (tuple(err_dim)))
    return err_ttl, err_dim

