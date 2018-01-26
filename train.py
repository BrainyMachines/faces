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
import gc
import sd_utils as utils
from copy import deepcopy
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.models as models
from torch.autograd import Variable
from models.resnet_feature_extractor import ResNetFeatureExtractor
import joblib
import faiss
# import pyflann as flann
import math

def main(args):

    args.dset_root = os.path.join(args.sm.scratch_dir, args.data, args.dset_name)
    args.gen_root = os.path.join(args.sm.scratch_dir, args.gen)
    args.ckpt_dir = os.path.join(args.gen_root, 'ckpt')
    utils.mkdir_p(args.gen_root)
    utils.mkdir_p(args.ckpt_dir)
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
    model_fe = ResNetFeatureExtractor(model)
    model_fe.train(False)

    args.model_file = os.path.join(args.ckpt_dir, 'fv_{:s}_{:s}.pth'.format(args.dset_name, args.arch))
    for epoch in range(args.num_epochs):
        logging.error('Epoch {:04d}'.format(epoch))
        embedding_id = [np.zeros((0, args.feature_dim), dtype=np.float32) for c in range(args.num_identities)]
        embedding_neg_id = [np.zeros((0, args.feature_dim), dtype=np.float32) for c in range(args.num_identities)]
        index_id = [[] for c in range(args.num_identities)]
        index_neg_id = [[] for c in range(args.num_identities)]
        embedding = np.zeros((len(train_dset), args.feature_dim), dtype=np.float32)
        for i, mb in enumerate(train_loader):
            img, target = mb
            fv = model_fe(Variable(img.cuda()))
            embedding[i*args.batch_size:(i+1)*args.batch_size,:] = fv.clone().data.cpu().numpy()
            bs = fv.size(0)
            for j in range(bs):
                # embedding_id[target[j]] = np.vstack((embedding_id[target[j]], fv[j,:].clone().data.cpu().numpy()))
                for c in range(args.num_identities):
                    if c != target[j]:
                        embedding_neg_id[c] = np.vstack((embedding_neg_id[c], fv[j,:].clone().data.cpu().numpy()))
                        index_neg_id[c].append(i * args.batch_size + j)
                    else:
                        embedding_id[target[j]] = np.vstack((embedding_id[target[j]], fv[j,:].clone().data.cpu().numpy()))
                        index_id[c].append(i * args.batch_size + j)

            fv = None
            img = None
            target = None
            mb = None
            gc.collect()
        joblib.dump(embedding, args.model_file)

        # hard = []    
        # ann = [flann.FLANN() for i in range(args.num_identities)]
        # ann_neg = [flann.FLANN() for i in range(args.num_identities)]
        # for k in range(args.num_identities):
        #     args.ann_file = os.path.join(args.ckpt_dir, 'ann_{:s}_{:s}_{:04d}.npz'.format(args.dset_name, args.arch, k))
        #     ann_params = ann[k].build_index(embedding_id[k], algorithm='autotuned', target_precision=0.9, log_level='error')
        #     ann_index, ann_dist = ann[k].nn_index(embedding_id[k], embedding_id[k].shape[0], checks=ann_params['checks'])
        #     ann_neg_params = ann_neg[k].build_index(embedding_neg_id[k], algorithm='autotuned', target_precision=0.9, log_level='error')
        #     ann_neg_index, ann_neg_dist = ann_neg[k].nn_index(embedding_id[k], args.num_neighbors, checks=ann_neg_params['checks'])
        #     for a_ in range(ann_index.shape[0]):
        #         for p_ctr in range(args.num_neighbors):
        #             p_ = int(ann_index.shape[1]) - 1  - p_ctr
        #             a = index_id[k][a_]
        #             for n_ in range(args.num_neighbors):
        #                 p = index_id[k][ann_index[a_, p_]]
        #                 n = index_neg_id[k][ann_neg_index[a_, n_]]
        #                 if ann_dist[a_, p_] - ann_neg_dist[a_, n_] + args.margin >= 0:  # hard example: violates margin 
        #                     hard.append((a, p, n))
        #     print('#Tuples: ', len(hard))
        #     print(hard)
        #     joblib.dump({'ann_index': ann_index, 'ann_dist': ann_dist, 'ann_neg_index': ann_neg_index, 'ann_neg_dist': ann_neg_dist}, args.ann_file)
        #     ann_params = None
        #     ann_index = None
        #     ann_dist = None
        # ann = None

        hard = []
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        ivfflat_config = faiss.GpuIndexIVFFlatConfig()
        ivfflat_config.device = 0
        # co = faiss.GpuClonerOptions()
        # co.useFloat16 = True
        for k in range(args.num_identities):
            d = embedding_id[k].shape[1]
            neg_nlist = int(math.sqrt( math.sqrt(embedding_neg_id[k].shape[0])))
            print(embedding_id[k].shape)
            print(embedding_neg_id[k].shape)
            print()
            
            # Build index
            index = None
            index = faiss.GpuIndexFlatL2(res,  d, flat_config)
            index.nprobe = args.nprobe_gpu_limit
            assert index.is_trained
            index.add(embedding_id[k])
            neg_index = None
            neg_index = faiss.GpuIndexIVFFlat(res, d, neg_nlist, faiss.METRIC_L2, ivfflat_config)
            # neg_index = faiss.GpuIndexFlatL2(res, d, flat_config)
            neg_index.nprobe = args.nprobe_gpu_limit
            assert not neg_index.is_trained
            neg_index.train(embedding_neg_id[k])
            assert neg_index.is_trained
            neg_index.add(embedding_neg_id[k])

            # Search
            ann_neg_dist, ann_neg_index = neg_index.search(embedding_id[k], args.num_neighbors)
            # print(ann_neg_dist)
            # print(ann_neg_index)
            ann_dist, ann_index = index.search(embedding_id[k], args.num_neighbors)
            # print(ann_index)

            # Generate hard triplets
            for a_ in range(ann_index.shape[0]):
                for p_ctr in range(args.num_neighbors):
                    p_ = int(ann_index.shape[1]) - 1  - p_ctr
                    a = index_id[k][a_]
                    for n_ in range(args.num_neighbors):
                        p = index_id[k][ann_index[a_, p_]]
                        n = index_neg_id[k][ann_neg_index[a_, n_]]
                        if ann_dist[a_, p_] - ann_neg_dist[a_, n_] + args.margin >= 0:  # hard example: violates margin 
                            hard.append((a, p, n))
            print('#Tuples: ', len(hard))

            index = None
            neg_index = None
            gc.collect()

        gc.collect()


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
                            help='image size (default: 224)')
        parser.add_argument('--feature_dim', default=512, type=int,
                            help='feature_dim (default: 512)')
        parser.add_argument('--num_channels', default=3, type=int,
                            help='num of channels 3: colored, 1: grayscale (default: 3)')
        parser.add_argument('--dset_name', type=str, default='celeb10',
                            help='dataset name (default:celeb10)')
        parser.add_argument('--batch_size', type=int, default=16,
                            help='minibatch size (default: 16')
        parser.add_argument('--num_epochs', type=int, default=100,
                            help='number of epochs (default: 100')
        parser.add_argument('--num_identities', type=int, default=10,
                            help='number of identities (default: 10')
        parser.add_argument('--num_neighbors', type=int, default=10,
                            help='number of neighbors (default: 10')
        parser.add_argument('--nprobe_gpu_limit', type=int, default=1024,
                            help='nprobe limit for faiss(default: 1024')
        parser.add_argument('--margin', type=float, default=0.2,
                            help='margin for triplet loss (default: 0.2')
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
