import random
import numpy as np
import os, datetime
import sys
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss

from losses import DiceLoss

from datasets.dataset_synapse_fold2 import Synapse_dataset_fold, RandomGenerator
from options.base_options import Base_options

from networks.DGC_seg_modeling import *

from utils.util import get_number_of_learnable_parameters, print_options

from inference import inference

if __name__ == "__main__":
    args = Base_options().parse()
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset

    dataset_config = {          # TODO dataset_config 
        'Synapse': {
            'Dataset': Synapse_dataset_fold,
            'train_path': 'F:/data/Synapse/vol_data',
            'test_volume_path':'F:/data/Synapse/vol_data',
            'save_base_path':'F:/data/Synapse/ginn_save', 
            'num_classes': 9,
            'z_spacing': 1,
        },
    }

    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.train_path = dataset_config[dataset_name]['train_path']
    args.test_volume_path = dataset_config[dataset_name]['test_volume_path']
    args.save_model_path = os.path.join(dataset_config[dataset_name]['save_base_path'], args.exp_name, "model")
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # check 
    args.exp = 'YG_' + dataset_name + str(args.img_size)      

    snapshot_path = os.path.join(args.save_model_path, args.exp, 'YG')                  
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs)                   
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)  
    snapshot_path = snapshot_path + '_imagesize'+str(args.img_size)
    snapshot_path = snapshot_path + '_seed'+str(args.seed)   
    
    now_time = datetime.datetime.now().strftime("%m-%d-%H-%M")
    snapshot_path = os.path.join(snapshot_path, now_time+"_"+args.dataset)

    model = ginn(3, args.inter_channels, args.num_classes).cuda()

    logging.basicConfig(filename=snapshot_path + "/log_run_" + args.exp_name + ".txt", level=logging.INFO,   # TODO logging file path
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    logging.info(f'{args.model_name}: Number of params: {sum([p.data.nelement() for p in model.parameters()])}')
    logging.info(f'Number of {args.model_name} learnable params: {get_number_of_learnable_parameters(model)}')

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # data 
    db_train = Synapse_dataset_fold(base_dir=args.train_path, n_fold=args.n_fold, fold_num=args.fold_num,dim=args.dim, split="train",random_seed=args.seed,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    logging.info("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    optimizer = optim.RMSprop(model.parameters(), lr=base_lr, weight_decay=1e-8, momentum=0.9)

    # TODO tensorboard files path
    writer = SummaryWriter(snapshot_path + '/log_run_' + args.exp_name)
    
    epoch_begin = 0
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1

    # TODO 若不存在the_best_model,则加载预训练模型数据, 若存在，则加载model中的数据继续训练
    if not os.path.exists(os.path.join(snapshot_path, 'best_model.pth')):
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
    else:
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
        state_dict = torch.load(os.path.join(snapshot_path, 'best_model.pth'))
        model.load_state_dict(state_dict['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(state_dict['optimizer'])
        epoch_begin = state_dict['epoch'] + 1
        iter_num = epoch_begin * len(trainloader)

    eval_score_higher_is_better = True
    if eval_score_higher_is_better:
        best_eval_score = float('-inf')
    else:
        best_eval_score = float('+inf')
    best_eval_dice = 0
    best_eval_HD95 = 100

    iterator = tqdm(range(epoch_begin, max_epoch), ncols=70)

    # train
    for epoch_num in iterator:
        iterator.set_description('Epoch %d' % epoch_num)
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(),loss_dice.item()))

            if iter_num % 20 == 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        test_interval = 1
        if epoch_num > int(max_epoch * 0.8) and (epoch_num + 1) % test_interval == 0:
            model.eval()
            with torch.no_grad():
                snapshot_name = snapshot_path.split('/')[-1]
                if args.is_savenii:
                    args.test_save_dir = os.path.join(dataset_config[dataset_name]['save_base_path'], args.exp_name, "predictions_run")
                    test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
                    os.makedirs(test_save_path, exist_ok=True)
                else:
                    test_save_path = None

                performance, mean_hd95, _ = inference(args, model, logging, test_save_path)

                # TODO remember best validation metric
                if eval_score_higher_is_better:
                    is_best = performance > best_eval_score
                else:
                    is_best = performance < best_eval_score

                if is_best:
                    logging.info('Saving new best evaluation metric: %f   mean_dice : %f   mean_hd95 : %f' % (performance, performance, mean_hd95))
                    best_eval_score = performance

                    best_eval_dice = performance
                    best_eval_HD95 = mean_hd95

                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save({
                        'model_state_dict':model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'epoch':epoch_num
                    }, save_mode_path)
                    logging.info("save the best model to {}".format(save_mode_path))

            model.train()
            logging.info('The best model of the current process:   mean_dice : %f   mean_hd95 : %f' % (best_eval_dice, best_eval_HD95))

        # TODO save model
        save_interval = 1 
        if epoch_num > int(max_epoch * 0.8) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save({
                        'model_state_dict':model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'epoch':epoch_num
                    }, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save({
                        'model_state_dict':model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'epoch':epoch_num
                    }, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
    
    logging.info('The best model of the whole process:   mean_dice : %f   mean_hd95 : %f' % (best_eval_dice, best_eval_HD95))
    writer.close()
    logging.info("Training Finished!") 