import os
import argparse
from pprint import pprint

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--data_rootdir',  type=str, default='./data/')
        self.parser.add_argument('--input',       type=str, default='gt', help='choises:{gt,cpn,sh}')
        self.parser.add_argument('--data_process', type=str, default='stat', help='choises:{stat,scale}. How to standardize input coordinates')
        self.parser.add_argument('--exp',            type=str, default='test')
        self.parser.add_argument('--ckpt',           type=str, default='checkpoint/')
        self.parser.add_argument('--test',           dest='test', action='store_true', help='evaluation phase')
        self.parser.add_argument('--stage', type=str, default='lifting', help='choices:{lifting, increment}')
        self.parser.add_argument('--num_jts',       type=int, default=17, help='regarded as inp_num_jts')
        self.parser.add_argument('--out_num_jts',       type=int, default=17)
        self.parser.add_argument('--gpu',           type=str, default='0', help='Only one GPU is enough to reach maximum training speed.')
        self.parser.add_argument('--eval_jt',   dest='eval_jt', action='store_true', help='evaluation on joint wise. default on action wise')
        self.parser.set_defaults(eval_jt=False)
        self.parser.add_argument('--train_sample',    type=int,   default=1, help='interval down sample train dataset. 1 means to keep as it is')
        self.parser.add_argument('--test_sample',    type=int,   default=1, help='interval down sample test dataset. 1 means to keep as it is')
        self.parser.add_argument('--use_data_cache', dest='use_data_cache', action='store_true', help='speed up data loading')
        self.parser.set_defaults(use_data_cache=False)
        self.parser.add_argument('--optimizer', type=str, default='adam')


        # ===============================================================x
        #                     Lifting model options
        # ===============================================================
        self.parser.add_argument('--lifting_model', type=str, default='linear', help='choices: {linear, lcn, semgcn}')
        self.parser.add_argument('--load',           type=str, default=None, help='checkpoint file path of lifting model')
        self.parser.add_argument('--hidsize',          type=int, default=1024, help='number of hidden node in nn.linear layer')
        self.parser.add_argument('--num_block',      type=int, default=2, help='number of residual blocks')
        self.parser.add_argument('--knn',       type=int, default=3)

        # ===============================================================
        #                     Increment model options
        # ===============================================================
        self.parser.add_argument('--load_inc',           type=str, default=None, help='checkpoint file path of residual regressors. inc means increment')
        self.parser.add_argument('--optim_pelvis',       dest='optim_pelvis', action='store_true', help='use lifting predict to optimize pelvis position')
        self.parser.set_defaults(optim_pelvis=False)
        self.parser.add_argument('--inc_hidsize',       type=int, default=256, help='number of hidden node in nn.linear layer')
        self.parser.add_argument('--inc_block',        type=int,  default=2, help='number of residual blocks')
        self.parser.add_argument('--inc_num',        type=int,  default=10, help='number of residual regressors')
        self.parser.add_argument('--inc_lr',       type=str,  default='0.001', help='eg. 0.001 | 0.001,0.0001,0.0001')
        self.parser.add_argument('--inc_epoch',     type=int,   default=1, help='training epoch for each residual regressor')
        self.parser.add_argument('--inc_input_type',     type=str,   default='delta', help='choices: {delta, cat, proj}')
        self.parser.set_defaults(proj_formula=False)
        self.parser.add_argument('--update_pelvis', dest='update_pelvis', action='store_true', help='update pelvis for each turn of ERD refinement')
        self.parser.set_defaults(update_pelvis=False)

        # ===============================================================
        #                     Training options
        # ===============================================================
        self.parser.add_argument('--max',            dest='max_norm', action='store_true', help='gradient clip')
        self.parser.set_defaults(max_norm=False)
        self.parser.add_argument('--lr',             type=float,  default=1.0e-3)
        self.parser.add_argument('--lr_decay',       type=int,    default=1, help='milestone epoch for lr decay')
        self.parser.add_argument('--lr_gamma',       type=float,  default=0.96, help='decay weight')
        self.parser.add_argument('--epoch',         type=int,    default=200)
        self.parser.add_argument('--dropout',        type=float,  default=0.25, help='dropout probability')
        self.parser.add_argument('--batch',    type=int,    default=200)
        self.parser.add_argument('--test_batch',    type=int,    default=1000)
        self.parser.add_argument('--loss',          type=str,   default='l2')
        self.parser.add_argument('--amsgrad',   dest='amsgrad', action='store_true', help='only for Adam optimizer')
        self.parser.set_defaults(amsgrad=False)
        self.parser.add_argument('-flip',   dest='flip',    action='store_true', help='data augmentation of horizontally flipping.')
        self.parser.set_defaults(flip=False)
        self.parser.add_argument('--test_flip', dest='test_flip',  action='store_true')
        self.parser.set_defaults(test_flip=False)

        self.parser.add_argument('--procrustes', dest='procrustes', action='store_true', help='use rigid aligment at testing')
        self.parser.set_defaults(procrustes=False)


        # ===============================================================
        #                     Feature options
        # ===============================================================
        self.parser.add_argument('--pca_input', dest='pca_input', action='store_true')
        self.parser.set_defaults(pca_input=False)
        self.parser.add_argument('--pca_component',  type=float, default=30, help='how many principle component should be extracted')


        # ===============================================================
        #                     Debug options
        # ===============================================================
        self.parser.add_argument('--debug_speedup', dest='debug_speedup', action='store_true')
        self.parser.set_defaults(debug_speedup=False)

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()

        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if not os.path.isdir(ckpt):
            os.makedirs(ckpt)
        self.opt.ckpt = ckpt
        self.opt.inc_lr = list(map(float, self.opt.inc_lr.split(',')))
        if len(self.opt.inc_lr) == 1:
            self.opt.inc_lr *= self.opt.inc_num
        if self.opt.stage == 'increment':
            self.opt.epoch = self.opt.inc_num * self.opt.inc_epoch
            self.lr_decay = 1
        if self.opt.gpu == '-1':
            self.opt.use_gpu = False
        else:
            self.opt.use_gpu = True
        self._print()

        return self.opt
