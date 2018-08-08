import numpy as np
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import keras

from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD




width = 224
height = 224
batch_size = 16

model = VGG16(include_top=True, weights=None)
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd) # loss='hinge'

x = np.zeros((batch_size, width, height, 3), dtype=np.float32)

y = np.zeros((batch_size, 1000), dtype=np.float32)

# warmup
model.train_on_batch(x, y)

t0 = time.time()
n = 0
while n < 100:
    tstart = time.time()
    model.train_on_batch(x, y)
    tend = time.time()
    print("Iteration: %d train on batch time: %7.3f ms." %( n, (tend - tstart)*1000 ))
    n += 1
t1 = time.time()

print("Batch size: %d" %(batch_size))
print("Iterations: %d" %(n))
print("Time per iteration: %7.3f ms" %((t1 - t0) *1000 / n))