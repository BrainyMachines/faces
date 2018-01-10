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
# import matplotlib as mpl
# mpl.use('Agg')
import torch
try:
    import ipdb as pdb
except:
    import pdb as pdb
sys.path.append('..')
import sd_utils as utils
# import face_alignment
import dlib
import skimage.io as skio
from scipy.misc import imresize
import cv2
import math
import pytest


def show_landmarks(img, shape, dets):
    win = dlib.image_window()
    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(shape)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()


def missing_equilateral_triangle_vertex(x1, y1, x2, y2, eps=1e-12):
    x1_, y1_, x2_, y2_ = 0.0, 0.0, 0.0, 0.0
    if x1 > x2:
        x1_, y1_, x2_, y2_ = x2, y2, x1, y1
    else:
        x1_, y1_, x2_, y2_ = x1, y1, x2, y2
    dx = x2_ - x1_
    dy = y2_ - y1_
    theta = math.pi / 3
    M = np.array([[math.cos(theta), -math.sin(theta), x1_], [math.sin(theta), math.cos(theta), y1_], [0.0, 0.0, 1.0]])
    V = np.array([[dx], [dy], [1.0]])
    W = np.dot(M, V)
    x3 = W[0, 0]
    y3 = W[1, 0]
    return x3, y3


def main(args):

    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=True, flip_input=False)
    left_eye_num = args.left_eye_num
    right_eye_num = args.right_eye_num
    width = args.imsize
    height = args.imsize
    out_pts = np.array([[0.3 * width, height / 3, 1.0],
                        [0.7 * width, height / 3, 1.0],
                        [0.5 * width, height / 3 + 0.4 * width * math.sin(math.pi / 3.0), 1.0]
                       ], dtype=np.float64)
    # In code sanity check for missing_equilateral_triangle_vertex 
    x, y = missing_equilateral_triangle_vertex(out_pts[0, 0], out_pts[0, 1], out_pts[1, 0], out_pts[1, 1])
    assert x == pytest.approx(out_pts[2, 0])                  
    assert y == pytest.approx(out_pts[2, 1])                  
    out_pts = out_pts.astype(np.uint8)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.landmarks_predictor)
    subsets = ['actors', 'actresses']
    for sdir in subsets:
        faces_dir = os.path.join(sm.scratch_dir, args.data, sdir, 'faces')
        for dentry in tqdm(os.scandir(faces_dir)):
            if dentry.is_dir():
                for fentry in os.scandir(dentry):
                    logging.debug(fentry.path + str(os.path.exists(fentry.path)))
                    img = None
                    img_ = None
                    try:
                        img_ = skio.imread(str(fentry.path))
                    except:
                        logging.error('imread failed: ' + str(fentry.path))
                        continue
                    if img_.shape[0] > img_.shape[1]:
                        x_sz = int(float(args.imsize) * float(img_.shape[0]) / float(img_.shape[1]))
                        img = imresize(img_, (x_sz, args.imsize), interp='bilinear', mode=None)
                    else:
                        y_sz = int(float(args.imsize) * float(img_.shape[1]) / float(img_.shape[0]))
                        img = imresize(img_, (args.imsize, y_sz), interp='bilinear', mode=None)
                    dets = detector(img)
                    if len(dets) == 1:
                        d = dets[0]
                        # logging.debug("{}/{} :: Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(, 
                        #                 dentry.name, fentry.name, k, d.left(), d.top(), d.right(), d.bottom()))
                        shape = predictor(img, d)
                        pts = np.zeros((3, 3), dtype=np.uint8)
                        left_point = shape.part(left_eye_num)
                        pts[0, 0] = np.uint8(left_point.x)
                        pts[0, 1] = np.uint8(left_point.y)
                        pts[0, 2] = np.uint8(1)
                        right_point = shape.part(right_eye_num)
                        pts[1, 0] = np.uint8(right_point.x)
                        pts[1, 1] = np.uint8(right_point.y)
                        pts[1, 2] = np.uint8(1)
                        x, y = missing_equilateral_triangle_vertex(left_point.x, left_point.y, right_point.x, right_point.y)
                        pts[2, 0], pts[2, 1], pts[2, 2] = np.uint8(x), np.uint8(y), np.uint8(1) 
                        # tform_ = cv2.estimateRigidTransform(pts, out_pts, False)
                        tform_ = dlib.find_affine_transform(pts, out_pts)
                        tform = np.vstack((tform_, np.array([[0.0, 0.0, 1.0]])))
                        landmarks = np.hstack((np.zeros((args.num_landmarks, 2)), np.ones((args.num_landmarks, 1))))
                        for i in range(args.num_landmarks):
                            shp = shape.part(i)
                            landmarks[i, 0] = shp.x
                            landmarks[i, 1] = shp.y
                        adjusted_landmarks = np.dot(landmarks, tform.T)
                        print(adjusted_landmarks) 
                    else:
                        logging.info('faces detected {:d}: '.format(len(dets)) + dentry.name + os.sep + fentry.name)
                    

                    # preds = fa.get_landmarks(im)
                    # logging.info(preds)


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
        parser.add_argument('--imsize', default=256, type=int,
                            help='image size (default: 256)')
        parser.add_argument('--num_channels', default=3, type=int,
                            help='num of channels 3: colored, 1: grayscale (default: 3)')
        parser.add_argument('--landmarks_predictor', required=True, type=str,
                            help='path to facial landmarks detector model')
        parser.add_argument('--num_landmarks', type=int, default=68,
                            help='number of facial landmarks')
        parser.add_argument('--left_eye_num', type=int, default=35,
                            help='index of left eye landmark [0-indexed] (default: 35)')
        parser.add_argument('--right_eye_num', type=int, default=44,
                            help='index of right eye landmark [0-indexed] (default: 44)')
    
        args = parser.parse_args()
        args.sm = sm


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