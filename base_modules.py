import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.decomposition import PCA
import copy
import tool.util as util
from dataset.human36m import Human36M, lcn_jt17_adj_idx, DataConverter, jt17_parent_idx
from dataset.human36m_openpose import Human36M_OpenPose
from dataset.human36m_cpn_op import Human36M_CPN_OP

def get_dataloader(opt, dataset=None, is_train=True, shuffle=True):
    if dataset is None:
        is_flip = opt.flip if is_train else opt.test_flip
        if not is_train and opt.stage != 'lifting' and (opt.input == 'sh' or opt.input == 'shft' or opt.input == 'cpn' or opt.input == 'openpose'):
            exclude_drift_data = True
        else:
            exclude_drift_data = False
        subdir = 'stat' if 'conf2d' not in opt.data_process else 'conf2d'
        actual_data_dir = os.path.join(opt.data_rootdir, opt.input, subdir)
        dataset_func = Human36M_OpenPose if opt.input == 'openpose' else Human36M
        if opt.pca_input:
            dataset_name = '' if 'humaneva' not in opt.data_rootdir else '_humaneva'
            if os.path.exists('pcamodel%d%s_%s_%s_trainset.pth.tar' % (int(opt.pca_component), dataset_name, opt.input, opt.data_process)):
                pca_model = torch.load('pcamodel%d%s_%s_%s_trainset.pth.tar' % (int(opt.pca_component), dataset_name, opt.input, opt.data_process))
                pca_model.components_gpu = torch.Tensor(pca_model.components_).cuda()
                pca_model.mean_gpu = torch.Tensor(pca_model.mean_).cuda()
                torch.save(pca_model, 'pcamodel%d%s_%s_%s_trainset.pth.tar' % (int(opt.pca_component), dataset_name, opt.input, opt.data_process))
            else:
                print('Generating PCA model...')
                opt_for_pca = copy.copy(opt)
                opt_for_pca.data_process = 'identity'
                opt_for_pca.pca_input = False
                data_converter = DataConverter(opt_for_pca)
                trainset_for_pca = dataset_func(data_path=actual_data_dir, is_train=True, flip=is_flip, exclude_drift_data=exclude_drift_data,
                               data_converter=data_converter, opt=opt_for_pca)
                pca_model = get_pca_model(trainset_for_pca, opt)
                pca_model.components_gpu = torch.Tensor(pca_model.components_).cuda()
                pca_model.mean_gpu = torch.Tensor(pca_model.mean_).cuda()
                if not opt.debug_speedup:
                    torch.save(pca_model, 'pcamodel%d_%s_%s_trainset.pth.tar' % (int(opt.pca_component), opt.input, opt.data_process))
        else:
            pca_model = None
        cache_fn = '%s_data_cache.pth.tar' % ('train' if is_train else 'test')
        if opt.use_data_cache and os.path.exists(os.path.join(opt.ckpt, cache_fn)):
            print('Use data cache %s' % os.path.join(opt.ckpt, cache_fn))
            dataset = torch.load(os.path.join(opt.ckpt, cache_fn))
        else:
            data_converter = DataConverter(opt)
            if 'cpn_op_cat' not in opt.exp:
                dataset = dataset_func(data_path=actual_data_dir, is_train=is_train, flip=is_flip, exclude_drift_data=exclude_drift_data,
                                   data_converter=data_converter, opt=opt, pca_model=pca_model)
            else:
                dataset = Human36M_CPN_OP(data_path_cpn=actual_data_dir, data_path_op=os.path.join(opt.data_rootdir, 'openpose', 'conf2d'), is_train=is_train, flip=is_flip, exclude_drift_data=exclude_drift_data,
                                   data_converter=data_converter, opt=opt, pca_model=pca_model)
            if opt.use_data_cache:
                torch.save(dataset, os.path.join(opt.ckpt, cache_fn))
                print('Stored data cache %s' % os.path.join(opt.ckpt, cache_fn))


    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=opt.batch if is_train else opt.test_batch,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True
    )
    return dataloader

def get_optimizer(model, opt):
    if opt.stage == 'increment':
        optimizer = []
        scheduler = []
        for i in range(opt.inc_num):
            if opt.optimizer == 'sgd':
                optimizer.append(torch.optim.SGD(model.model[i].parameters(), lr=opt.inc_lr[i]))
            else:
                optimizer.append(torch.optim.Adam(model.model[i].parameters(), lr=opt.inc_lr[i], amsgrad=opt.amsgrad))
            scheduler.append(torch.optim.lr_scheduler.StepLR(optimizer[i], step_size=2, gamma=opt.lr_gamma))
    else:
        if opt.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, amsgrad=opt.amsgrad)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt.lr_gamma)
    return optimizer, scheduler

def get_bi_optimiezer(model1, model2, opt):
    optimizer = torch.optim.Adam(itertool.chain(model1.parameters(), model2.parameters()), lr=opt.lr, amsgrad=opt.amsgrad)
    # optimizer = torch.optim.Adam([model1.parameters(), model2.parameters()], lr=opt.lr, amsgrad=opt.amsgrad)

def get_loss(opt):
    if opt.loss == 'l2':
        criterion = nn.MSELoss(reduction='mean').cuda()
    elif opt.loss == 'sqrtl2':
        criterion = lambda output, target: torch.mean(torch.norm(output - target, dim=-1))
    elif opt.loss == 'l1':
        criterion = nn.L1Loss(reduction='mean').cuda()
    else:
        raise Exception('Unknown loss type %s' % opt.loss)

    return criterion


def get_knn_idx(K, num_jts=17):
    assert num_jts == 17, 'num_jts is %d, not 17' % num_jts
    knn_mat = get_knn_mat(K, num_jts)
    knn_idx = []
    for i in range(num_jts):
        knn_idx.append(np.where(knn_mat[i]>0)[0].tolist())
    return knn_idx

def get_knn_mat(K, num_jts=17, adj_idx=lcn_jt17_adj_idx):
    assert num_jts == 17, 'num_jts is %d, not 17' % num_jts
    jt17_adj_mat = np.zeros([num_jts, num_jts])
    for i in range(num_jts):
        for j in adj_idx[i]:
            jt17_adj_mat[i, j] = 1
    knn_adj_mat = (np.linalg.matrix_power(jt17_adj_mat, K) > 0).astype(int)
    return knn_adj_mat

def get_pca_model(dataset, opt):
    data_converter = DataConverter(opt)
    data = np.array(dataset.inp)
    data_normed = data_converter.normalize(data)
    n_components = opt.pca_component

    num_samples = len(data)
    print(">>> Generating PCA model on %s train dataset with %d samples..." % (opt.input, num_samples))
    data_normed = data_normed.reshape(num_samples, -1)
    if n_components > 1:
        n_components = int(n_components)
    pca = PCA(n_components=n_components)
    model = pca.fit(data_normed)
    recovery_ratio = sum(model.explained_variance_ratio_)
    print("Maintain %f%% variance with %d components" % (recovery_ratio*100, model.n_components_))
    return model

