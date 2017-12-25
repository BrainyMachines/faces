from __future__ import print_function
import os
import errno
#import psutil
import numpy as np
import numpy
from PIL import Image
#import subprocess32 as subprocess
import subprocess
import hashlib
import socket
import re
from datetime import datetime
import sys
import warnings
from copy import deepcopy
from pprint import pprint   
try:
    from os import scandir
except:
    from scandir import scandir
import torch


def mkdir_p(path):
    """ Equivalent of mkdir -p """
    """ source: http://bit.ly/1dyli3d """
    try:
        os.makedirs(path)
    except OSError as exc:   # Python >2.5
        pass 
        # if exc.errno == errno.EEXIST and os.path.isdir(path):
        #    pass
        # else:
        #    raise


# def memory_usage():
#     """ return the memory usage in MB """
#     """ source: http://bit.ly/1dspz7I """
#     process = psutil.Process(os.getpid())
#     try:
#         mem = process.get_memory_info()[0] / float(2 ** 20)
#     except:
#         mem = process.memory_info()[0] / float(2 ** 20)
#     return mem


def wc_l(fname):
    """ return number of lines in a file """

    lineCount = 0
    try:
        with open(fname, 'r') as f:
            for line in f:
                lineCount = lineCount + 1
    except:
        print('Could not open file ', fname)
        pass
    return lineCount


def git_version():
    """ returns git revision """
    """ source: http://bit.ly/1Ctm1ho """

    from subprocess import Popen, PIPE
    gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout=PIPE)
    (stdout, _) = gitproc.communicate()
    return stdout.strip()

def images_from_batch(xgen, rows=16, cols=16, order='bcwh'):

        I = np.round(255 * xgen).astype(np.float32)
        if order == 'bcwh':
            I = np.transpose(I, (0,2,3,1))
        bs = I.shape[0]
        im_rows = I.shape[1]
        im_cols = I.shape[2]
        im_channels = I.shape[3]
        im = np.zeros((rows*im_rows, cols*im_cols, im_channels))
        c = 0
        for i in range(rows):
            for j in range(cols):
                if c >= bs:
                    break
                im[i*im_rows:(i+1)*im_rows, j*im_cols:(j+1)*im_cols, :] = I[c,:,:,:]
                c = c + 1
        return im.astype(np.uint8)


def todict(obj, classkey=None):
    """ convert object to dict """
    """ source: http://bit.ly/1ZM6Acc  """

    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = todict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return todict(obj._ast())
    elif hasattr(obj, "__iter__"):
        return [todict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, todict(value, classkey))
                    for key, value in obj.__dict__.iteritems()
                    if not callable(value) and not key.startswith('_')])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj


def save_obj(obj, fname):
    """ save object to npz file """

    obj_dict = todict(obj)
    np.savez(fname, **obj_dict)


def load_obj(obj, fname):
    """ load variables from npz file """

    npzfile = np.load(fname)
    for k in npzfile.files:
        setattr(obj, k, npzfile[k])
    return obj


def sha1_hash(fname, blocksize=4096):
    """ compute sha1hash of a file """
    hash = ''
    if not os.path.exists(fname):
        errmsg = "File %s does not exist" % (fname)
        print(errmsg)
        return ''
    try:
        hasher = hashlib.sha1()
        with open(fname, 'rb') as f:
            buf = f.read(blocksize)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(blocksize)
        hash = hasher.hexdigest()
    except:
        print("Exception in hashing file")
        raise
    return hash


def rsync_and_verify(src, dst, verify=False, max_attempts=1):
    """Rsync src to dst and verify if copy is done"""

    print('Rsync %s to %s on %s\n' % (src,
                                      dst,
                                      socket.gethostname()))
    sys.stdout.flush()
    src_ = deepcopy(src)
    dst_ = deepcopy(dst)
    src_cred = ''
    src_path = ''
    dst_cred = ''
    dst_path = ''
    rsync_path = ''

    if ':' in src:
        src_cred, src_path = src.split(':')
    else:
        src_cred = ''
        src_path = src

    if ':' in dst:
        dst_cred, dst_path = dst.split(':')
    else:
        dst_cred = ''
        dst_path = dst

    if src_cred == '':
        mkdir_p(src_path)
    else:
        rsync_path = '--rsync-path=' + '"' + 'mkdir -p' + ' ' + src_path + ' ' + '&&' + ' ' + 'rsync' + '"'
    
    if dst_cred == '':
        mkdir_p(dst_path)
    else:
        rsync_path = '--rsync-path=' + '"' + 'mkdir -p' + ' ' + src_path + ' ' + '&&' + ' ' + 'rsync' + '"'

    if src_[-1] != os.sep:
        src_ = src_ + os.sep
    
    if dst_[-1] != os.sep:
        dst_ = dst_ + os.sep

    for attempt in range(max_attempts):
        print('attempt %d' % attempt)
        try:
            copycmd = 'rsync -av' + ' ' + rsync_path + ' ' + src_ + ' ' + dst_ 
            pprint(copycmd)
            sys.stdout.flush()
            output = subprocess.check_output(copycmd,
                                             shell=True)
            pprint(output)
            sys.stdout.flush()

            if verify:
                # Verify if the copying is done correctly
                if os.path.isdir(src):
                    for fl in os.listdir(src):
                        sfile = src + os.sep + fl
                        dfile = dst + os.sep + fl
                        shash = sha1_hash(sfile)
                        dhash = sha1_hash(dfile)
                        if shash != dhash:
                            print('Hashes of files %s and %s do not match.' % (sfile, dfile))
                            print('Error in copying. Quitting ...\n')
                            sys.stdout.flush()
                            raise Exception('hash mismatch')
                        print('.', end='')
                        sys.stdout.flush()
                else:
                    shash = sha1_hash(src)
                    dhash = sha1_hash(dst)
                    if shash != dhash:
                        print('Hashes of files %s and %s do not match.' % (src, dst))
                        print('Error in copying. Quitting ...\n')
                        sys.stdout.flush()
                        raise Exception('hash mismatch')
                print('Hash check passed')
                sys.stdout.flush()

            break    # break if successful
        # except Exception, arg:
        except:
            # print('Error:', arg)
            print('Error in rsync')
            pass     # else retry

class StorageManager:

    def __init__(self, user='sourabhd', server='localhost', perm_storage='/data', scratch_storage='/tmp'):
        self.user = user
        self.server = server
        self.perm_storage = perm_storage
        self.scratch_storage = scratch_storage
        self.perm_dir = self.perm_storage + os.sep + self.user
        self.scratch_dir = self.scratch_storage + os.sep + self.user
        mkdir_p(self.perm_dir)
        mkdir_p(self.scratch_dir)

    def server_to_compute(self, src, dst):
        src_ = deepcopy(src)
        dst_ = deepcopy(dst)
        src_ = self.perm_dir + os.sep + src_ 
        dst_ = self.scratch_dir + os.sep + dst_
        src_ = self.user + '@' + self.server + ':' + src_
        rsync_and_verify(src_, dst_)

    def compute_to_server(self, src, dst):
        src_ = deepcopy(src)
        dst_ = deepcopy(dst)
        src_ = self.scratch_dir + os.sep + src_
        dst_ = self.perm_dir + os.sep + dst_
        dst_ = self.user + '@' + self.server + os.sep + ':' + dst_
        rsync_and_verify(src_, dst_)

def count_files(dir):
    return len([1 for x in list(scandir(dir)) if x.is_file()])


def PIL2array(img):
    """Source: http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/ """
    return numpy.array(img.getdata(),
                    numpy.uint8).reshape(img.size[1], img.size[0], 3)


def array2PIL(arr, size):
    """Source: http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/ """
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = numpy.c_[arr, 255*numpy.ones((len(arr),1), numpy.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

def arr2str(arr, max_line_width=250, precision=4):
    """Print numpy array, torch tensor, or variable"""
    np.set_printoptions(threshold=np.nan, linewidth=max_line_width, precision=precision)
    str_ = ''
    if isinstance(arr, torch.autograd.Variable):
        arr2str(arr.data.clone()    , max_line_width=max_line_width, precision=precision)    
    elif isinstance(arr, torch.CudaFloatTensorBase):
        str_ = np.array2string(arr.clone().cpu().numpy(), max_line_width=max_line_width, precision=precision)
    elif isinstance(arr, torch.Tensor):
        str_ = np.array2string(arr.clone().numpy(), max_line_width=max_line_width, precision=precision)
    elif isinstance(arr, np.ndarray):
        str_ = np.array2string(arr, max_line_width=max_line_width, precision=precision)
    elif isinstance(arr, list):
        str_ = '\n'.join([arr2str(a, max_line_width, precision) for a in arr])
    else:
        str_ = ''
    return str_

################################################################################
## Early stopping code adapted for PyTorch from Keras
## https://keras.io/callbacks/#earlystopping
## 

class EarlyStopping:
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """

    def __init__(self, monitor='val_loss',
                 min_delta=0, patience=0, verbose=0, mode='auto'):

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, current):
        stop = False
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                stop = True
            self.wait += 1
        return stop

    def on_train_end(self):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))


def get_hostname_timestamp_id():
    return socket.gethostname() + '_' + re.sub(r'\W+', '', str(datetime.now()))

def nested_setattr(obj, name_list, value):
    assert len(name_list) >= 1
    if len(name_list) == 1:
        setattr(obj, name_list[0], value) 
    else:
        nested_setattr(getattr(obj, name_list[0]), name_list[1:], value)

