import numpy as np
import os
import shutil

import torch
import torch.nn as nn

def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def print_options(opt, logging):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        # comment = ''
        # default = self.parser.get_default(k)
        # if v != default:
        #     comment = '\t[default: %s]' % str(default)
        # message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '------------------- End -----------------\n'
    logging.info(message)