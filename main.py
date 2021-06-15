from __future__ import absolute_import

import os
import time
import sys
import numpy as np
sys.path.append('./')
import tool.log as log
from argument import Options
import base_modules
from lifting_stage import train_lifting, test_lifting, get_lifting_model
from increment_stage import prepare_increment, train_increment, test_increment, get_increment_model, test_proj, inference_increment, update_pelvis_position
from base_modules import get_dataloader, get_optimizer, get_loss
import torch
import pdb

# import pydevd_pycharm
# pydevd_pycharm.settrace('10.249.173.77', port=12344, stdoutToServer=True, stderrToServer=True, suspend=False)

def main(opt):
    if opt.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        torch.multiprocessing.set_sharing_strategy('file_system')
    date = time.strftime("%y_%m_%d_%H_%M", time.localtime())
    log.save_options(opt, opt.ckpt)

    print(">>> Loading dataset...")
    if not opt.test:
        train_loader = get_dataloader(opt, is_train=True, shuffle=True)
    test_loader = get_dataloader(opt, is_train=False, shuffle=False)

    lifting_model = get_lifting_model(opt)
    if opt.stage == 'increment':
        increment_model = get_increment_model(opt)
        # 1. infer lifting result in advance
        # 2. modify pelvis if pelvis_model exists
        if not opt.test:
            train_loader_ordered = get_dataloader(opt, train_loader.dataset, is_train=True, shuffle=False)
            prepare_increment(train_loader_ordered, lifting_model, opt)
        prepare_increment(test_loader, lifting_model, opt)
    else:
        increment_model = None
    criterion = get_loss(opt)

    if opt.test:
        # test log
        logger = log.Logger(os.path.join(opt.ckpt, 'inference-%s.txt' % date))
        logger.set_names(['loss_test', 'err_test', 'err_x', 'err_y', 'err_z'])
        # test lifting
        print(">>> Test lifting<<<")
        # test lifting
        loss_test, err_test, err_dim = test_lifting(-1, test_loader, lifting_model, criterion, opt)
        logger.addmsg('lifting')
        logger.append([loss_test, err_test, err_dim[0], err_dim[1], err_dim[2]],
                      ['float', 'float', 'float', 'float', 'float'])
        if opt.stage == 'increment':
            # test increment
            if opt.proj_formula:
                print(">>> Test projection formula <<<")
                err_proj, err_proj_dim = test_proj(test_loader, opt)
                logger.addmsg('Projection Formula')
                logger.append([0, err_proj, err_proj_dim[0], err_proj_dim[1], err_proj_dim[2]],
                              ['float', 'float', 'float', 'float', 'float'])
            else:
                inc_num = len(increment_model.model)
                print(">>> Test increment (best at inc %d)<<<" % inc_num)
                loss_inc, err_inc, err_inc_dim = test_increment(-1, test_loader, increment_model.model, criterion, opt)
                logger.addmsg('increment')
                logger.append([loss_inc, err_inc, err_inc_dim[0], err_inc_dim[1], err_inc_dim[2]],
                              ['float', 'float', 'float', 'float', 'float'])
        elif opt.stage == 'lifting':
            # have finished executation
            pass
        else:
            raise Exception('Unknown opt.stage %s' % opt.stage)
        sys.exit()
    else:
        logger = log.Logger(os.path.join(opt.ckpt, 'log-%s.txt' % date))
        logger.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'err_test', 'err_x', 'err_y', 'err_z'])
        if opt.stage == 'lifting':
            optimizer, scheduler = get_optimizer(lifting_model, opt)
        elif opt.stage == 'increment':
            optimizer, scheduler = get_optimizer(increment_model, opt)
            loss_test, err_test, err_dim = test_lifting(-1, test_loader, lifting_model, criterion, opt)
            logger.addmsg("lifting")
            logger.append([0, 0, 0, loss_test, err_test, err_dim[0], err_dim[1], err_dim[2]],
                          ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float'])
        else:
            raise Exception('Unknown opt.stage %s' % opt.stage)
    inc_nth = -1
    err_best = np.inf
    for epoch in range(opt.epoch):
        # prepare current increment
        if opt.stage == 'increment':
            if epoch % opt.inc_epoch == 0:
                inc_nth += 1
                print('Start training increment %d' % inc_nth)
                logger.addmsg('increment %d' % inc_nth)
        lr_now = scheduler[inc_nth].get_lr()[0] if isinstance(scheduler, list) else scheduler.get_lr()[0]
        print('==========================')
        print('>>> epoch: {} | lr: {:.8f}'.format(epoch + 1, lr_now))

        if opt.stage == 'lifting':
            loss_train = train_lifting(epoch, train_loader, lifting_model, criterion, optimizer, opt)
            loss_test, err_test, err_dim = test_lifting(epoch, test_loader, lifting_model, criterion, opt)
        elif opt.stage == 'increment':
            loss_train = train_increment(epoch, train_loader, increment_model.model[:inc_nth + 1], criterion, optimizer[inc_nth], opt)
            loss_test, err_test, err_dim = test_increment(epoch, test_loader, increment_model.model[:inc_nth + 1], criterion, opt)

            if opt.optim_pelvis and opt.update_pelvis:
                _, predict_train, _, _ = inference_increment(train_loader_ordered, increment_model.model[:inc_nth + 1], criterion, opt)
                update_pelvis_position(train_loader_ordered, opt, predict_pose3d=predict_train)
                _, predict_test, _, _ = inference_increment(test_loader, increment_model.model[:inc_nth + 1], criterion, opt)
                update_pelvis_position(test_loader, opt, predict_pose3d=predict_test)

        else:
            raise Exception()

        if epoch % opt.lr_decay == 0:
            if opt.stage == 'lifting':
                scheduler.step()
            else:
                scheduler[inc_nth].step()

        logger.append([epoch+1, lr_now, loss_train, loss_test, err_test, err_dim[0], err_dim[1], err_dim[2]],
                      ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float'])

        # save ckpt
        is_best = err_test < err_best
        err_best = min(err_test, err_best)
        stored_model_weight = lifting_model.state_dict() if opt.stage == 'lifting' else increment_model.state_dict()
        log.save_ckpt({'epoch': epoch+1,
                       'lr': lr_now,
                       'error': err_best,
                       'inc_num': inc_nth+1,
                       'state_dict': stored_model_weight,
                       'optimizer': optimizer},
                      ckpt_path=opt.ckpt,
                      is_best=is_best)
    logger.close()

if __name__ == '__main__':
    opt = Options().parse()

    main(opt)
