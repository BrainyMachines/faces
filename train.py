

import os
import sys
import argparse
import time
import platform
from datetime import datetime
import logging
import numpy as np
try:
    import ipdb as pdb
except:
    import pdb as pdb
import sd_utils as utils
from copy import deepcopy
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.models as models


def main(args):

    args.dset_root = os.path.join(args.sm.scratch_dir, args.data, args.dset_name)
    # print(arg.dset_root, os.path.exists(args.dset_root))

    # Transforms
    transforms_train = transforms.Compose([
                        transforms.Resize(args.imsize),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                      ])        

    transforms_val = transforms.Compose([
                        transforms.Resize(args.imsize),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                      ])        

    # Dataset
    args.dset_root_train = os.path.join(args.dset_root, 'train')
    train_dset = datasets.ImageFolder(args.dset_root_train, transforms_train)

    args.dset_root_val = os.path.join(args.dset_root, 'val')
    val_dset = datasets.ImageFolder(args.dset_root_val, transforms_val)

    # Data Loader 
    train_loader = data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = data.DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Uncomment to test data loader
    # itr = iter(train_loader)
    #img, target = next(itr)
    # print(img)
    # print(target)    
    # pdb.set_trace()

    # Model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model = model.cuda()



if __name__ == '__main__':

    runinfo = {}
    runinfo['platform'] = platform.uname()
    runinfo['git_rev'] = utils.git_version()
    runinfo['start_time'] = datetime.now()
    runinfo['end_time'] = None
    runinfo['params'] = {}
    print('Platform: ', runinfo['platform'])
    print('Git revision:', runinfo['git_rev'])
    print('Start Time: ', runinfo['start_time'])
    sys.stdout.flush()
    start_time = time.time()
    start_ctime = time.clock()
    try:
        sm = utils.StorageManager()


        model_names = sorted(name for name in models.__dict__
            if name.islower() and not name.startswith("__")
            and callable(models.__dict__[name]))


        parser = argparse.ArgumentParser(description='Face verification training code')
        parser.add_argument('--data', type=str, default='datasets/facescrub_images',
                            help='input data directory')
        parser.add_argument('--data_perm', type=str, default='datasets/facescrub_images',
                            help='output data directory')
        parser.add_argument('--gen', type=str, default='gen/facescrub_images',
                            help='directory for generated files')
        parser.add_argument('--gen_perm', type=str, default='gen/facescrub_images',
                            help='directory for generated files on permanent storage')
        parser.add_argument('--random_seed', type=int, default=12345,
                            help='random seed')
        parser.add_argument('--job', type=str, default=utils.get_hostname_timestamp_id(),
                            help='unique identifier for job')
        parser.add_argument('--gpuid', type=int, default=0,
                            help='id of the GPU to run job on')
        parser.add_argument('--num_workers', default=8, type=int,
                            help='number of worker processes (defult: 8)')
        parser.add_argument('--imsize', default=224, type=int,
                            help='image size (default: 256)')
        parser.add_argument('--num_channels', default=3, type=int,
                            help='num of channels 3: colored, 1: grayscale (default: 3)')
        parser.add_argument('--dset_name', type=str, default='celeb10',
                            help='dataset name (default:celeb10)')
        parser.add_argument('--batch_size', type=int, default=16,
                            help='minibatch size (default: 16')
        parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',choices=model_names,
                            help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
      
        args = parser.parse_args()
        args.sm = sm

        logger = logging.getLogger("../logs/faces_train.log")
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        logger = logging.getLogger("../../logs/face_alignment.log")
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        sm.server_to_compute(args.data_perm, args.data)

        main(args)

        end_time = time.time()
        end_ctime = time.clock()
        print('Success: wall time:  %f sec, processor time: %f sec'
              % (end_time-start_time, end_ctime-start_ctime))
        runinfo['end_time'] = datetime.now()
        print('End Time: ', runinfo['end_time'])
        sys.stdout.flush()
    except:
        end_time = time.time()
        end_ctime = time.clock()
        print('Failure: wall time: %f sec, processor time: %f sec' %
              (end_time-start_time, end_ctime-start_ctime))
        runinfo['end_time'] = datetime.now()
        print('End Time: ', runinfo['end_time'])
        logging.exception(sys.exc_info()[2])
        pdb.post_mortem(sys.exc_info()[2])
