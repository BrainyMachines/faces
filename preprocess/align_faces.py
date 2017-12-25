import os
import sys
import argparse
import numpy as np
import time
import platform
from datetime import datetime
from tqdm import tqdm
import gc
import logging
import torch
import skimage.io as skio
try:
    import ipdb as pdb
except:
    import pdb as pdb

sys.path.append('..')
import sd_utils as utils

# import face_alignment


def main(args):
    print(args)     


if __name__ == '__main__':

    print(sys.path)
    runinfo = {}
    runinfo['platform'] = platform.uname()
    runinfo['git_rev'] = utils.git_version()
    runinfo['start_time'] = datetime.now()
    runinfo['end_time'] = None
    runinfo['params'] = {}
    print('Platform: ', runinfo['platform'])
    print('Numpy:', np.__version__)
    print('PyTorch', torch.__version__)
    print(torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor(1)))
    cudnn_version = torch.backends.cudnn.version()
    print('CUDNN VERSION', torch.backends.cudnn.version())
    print('Git revision:', runinfo['git_rev'])
    print('Start Time: ', runinfo['start_time'])
    start_time = time.time()
    start_ctime = time.clock()
    try:
        sm = utils.StorageManager()

        parser = argparse.ArgumentParser(description='Align, crop and select faces for training')


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
        parser.add_argument('--imsize', default=96, type=int,
                            help='image size (default: 96)')
        parser.add_argument('--num_channels', default=3, type=int,
                            help='num of channels 3: colored, 1: grayscale (default: 3)')
    
        args = parser.parse_args()
        args.sm = sm

        logger = logging.getLogger("../logs/face_alignment.log")
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