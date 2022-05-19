import argparse
import pickle
import torch
import os

class Base_options():
    def __init__(self) -> None:
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--exp_name', type=str, default='GCNUNET_test',              # TODO exp_name
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--model_name', type=str,
                    default='gcn_inn_skip3', help='select one unet model[gcnunet, dgcunet]')

                    
        parser.add_argument('--train_path', type=str,
                    default='', help='train dir for data')
        parser.add_argument('--test_volume_path', type=str,
                    default='', help='test dir for data')
        parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name Synapse candi')
        parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
        
        parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
        parser.add_argument('--max_epochs', type=int,
                    default=500, help='maximum epoch number to train')
        parser.add_argument('--batch_size', type=int,                                           # TODO Batch size
                    default=16, help='batch_size per gpu')

        parser.add_argument('--n_fold', type=int,                                           # TODO Batch size
                    default=5, help='n_fold')
        parser.add_argument('--fold_num', type=int,                                           # TODO Batch size
                    default=0, help='select fold number 0-n_fold')
        parser.add_argument('--dim', type=int,                                           # TODO Batch size
                    default=2, help='2 dim or 3 dim')


        parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')                   # TODO GPU numbers
        parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
        parser.add_argument('--base_lr', type=float,  default=2e-5,         #0.01
                    help='segmentation network learning rate')
        parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
        parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
        
        parser.add_argument('--is_savenii', action="store_false", help='whether to save results during inference')

        parser.add_argument('--inter_channels', type=int,
                    default=16, help='inter channels')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        # chech if it has been initialized
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        opt = parser.parse_args()
        # save and return the parser
        self.parser = parser
        return opt
    
    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '------------------- End -----------------\n'
        print(message)

    def parse(self):
        opt = self.gather_options()
        # self.print_options(opt)

        self.opt = opt
        return self.opt
