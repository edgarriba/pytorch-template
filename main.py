# system
import os
import cv2
import logging
import argparse
import numpy as np

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# NOTE: use torch.utils.tensorboard in future
from tensorboardX import SummaryWriter

# custom
from train import train
from validate import validate
from utils import create_directory
from model import MyModel
from losses import MyLoss
from dataset import MyDataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def TrainingApp():
    # Training settings
    parser = argparse.ArgumentParser(description='Training application.')
    # Data parameters
    parser.add_argument('--input-dir', type=str, required=True,
                        help='the path to the directory with the input data.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='the path to output the results.')
    parser.add_argument('--output-dir-val', type=str, required=True,
                        help='the path to output the results for validation.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # Optimization parameters
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd', 'rmsprop'], default='adam',
                        help='the optimization solver to use (default: adam)')
    parser.add_argument('--num-epochs', type=int, default=50, metavar='N',
                        help='number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=1e-5, metavar='WD',
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--batch-size', type=int, default=2, metavar='B',
                        help='the mini-batch size (default: 2)')
    # Training parameters
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=666, metavar='S',
                        help='random seed (default: 666)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=50,
        metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument(
        '--log-interval-vis',
        type=int,
        default=5000,
        metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='use tensorboard for logging purposes')
    parser.add_argument('--num-workers', default=8, type=int,
                        help='the number of workers for the dataloader.')
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()

    tb_writer = None
    if args.tensorboard:
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    # parse device string to a int ids
    device_ids = list(map(int, args.gpu_id.replace(',', '').strip()))
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type + ':' + str(device_ids[0]))
    args.device = device
    logger.info('=> device-ids: %s' % device_ids)
    logger.info('=> device-type: %s' % device_type)
    logger.info('=> primary device: %s' % device)

    # set the random seed
    np.random.seed(args.seed)
    cv2.setRNGSeed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        cudnn.enabled = True
        cudnn.benchmark = True
        torch.cuda.set_device(device_ids[0])
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    create_directory(args.output_dir)
    create_directory(args.output_dir_val)

    # create the dataset loader for training
    data_generator_train = MyDataset(
        data_root=args.input_dir,
        mode='train')

    # data loader for validation
    data_generator_val = MyDataset(
        data_root=args.input_dir,
        mode='val')


    data_loader_train = DataLoader(data_generator_train,
                                   shuffle=True,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   pin_memory=True)

    data_loader_val = DataLoader(data_generator_val,
                                 shuffle=False,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 pin_memory=True)

    # create model
    model = MyModel(num_outputs=1)
    model = model.to(device)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), \
            "Invalid file {}".format(args.resume)
        logger.info('=> loading checkpoint {}'.format(args.resume))
        pretrained_dict = torch.load(args.resume)
        net.load_state_dict(pretrained_dict, strict=False)
    else:
        logger.info('=> no checkpoint found at {}'.format(args.resume))

    # in case we want to use multiple devices for training
    model = nn.DataParallel(model, device_ids=device_ids)

    criterion = MyLoss()

    # create the optimizer the learning rate scheduler
    params = model.parameters()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.lr,
                               weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=args.lr,
                                  weight_decay=args.wd)
    else:
       raise NotImplementedError("Not supported solver {}".format(args.optimizer))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # epochs loop

    for epoch in range(args.num_epochs):

        train(epoch, model, optimizer, criterion, data_loader_train, tb_writer,
              args)

        with torch.no_grad():
            validate(epoch, model, criterion, data_loader_val, tb_writer, args)

        try:
            model_state_dict = model.module.state_dict()
        except:
            model_state_dict = model.state_dict()

        torch.save(model_state_dict, os.path.join(
            args.output_dir, '{0}_model.pth'.format(epoch)))

        # Update lr scheduler
        scheduler.step()


if __name__ == '__main__':
    TrainingApp()