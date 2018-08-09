'''
GPU benchmark script

Model used VGG16, Keras

Optional parameters
--n_iter: Number of iterations
--batch_size
--gpus: Number of GPUs

'''





import numpy as np
import time

import os

import keras

from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from keras.utils import multi_gpu_model

import argparse

parser = argparse.ArgumentParser(description='Benchmark arguments')
parser.add_argument('--n_iter', type=int, default=100, help='Number of iterations' )
parser.add_argument('--batch_size', type=int, default=16,
		    help='batch_size, if low -> fewer operations per iteration')
parser.add_argument('--gpus', type=int, help='Number of gpus to use', default=1)
args = parser.parse_args()


n_iter = args.n_iter
gpus = args.gpus

width = 224
height = 224
batch_size = args.batch_size

model = VGG16(include_top=True, weights=None)
if (gpus>=2):
    model = multi_gpu_model(model, gpus=gpus)
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd) # loss='hinge'

x = np.zeros((batch_size, width, height, 3), dtype=np.float32)

y = np.zeros((batch_size, 1000), dtype=np.float32)

# warmup
model.train_on_batch(x, y)

t0 = time.time()
n = 0
while n < n_iter:
    tstart = time.time()
    model.train_on_batch(x, y)
    tend = time.time()
    print("Iteration: %d train on batch time: %7.3f ms." %( n, (tend - tstart)*1000 ))
    n += 1
t1 = time.time()

print("Number of GPUs: %d" %(gpus))
print("Batch size: %d" %(batch_size))
print("Iterations: %d" %(n))
print("Time per iteration: %7.3f ms" %((t1 - t0) *1000 / n))
