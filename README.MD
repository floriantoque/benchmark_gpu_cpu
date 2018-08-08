## Benchmark GPU vs CPU (Python 3)

Machine 1  
  * GPU: GEFORCE GTX 1080 Ti (12GB)    
  * CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz  
  * Python 3.5, Tensorflow version : 1.9.0, Numpy version: 1.13.1  
          
Machine 2  
  * GPU: Tesla V100 (32GB)    
  * CPU: Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz  
  * Python 3.6.5, Tensorflow version : 1.8.0, Numpy version: 1.14.3



### 1) Matrix multiplication
Python script : matmul.py

10 iterations  
Matrix1 : shape(8192,8192), values 1  
Matrix2 : shape(8192,8192), values 1  

Machine 1  
GPU: 8192 x 8192 matmul took: 0.23 sec, 4805.72 G ops/sec  
CPU: 8192 x 8192 matmul took: 5.75 sec, 191.34 G ops/sec  

Machine 2  
GPU: 8192 x 8192 matmul took: 0.08 sec, 13581.87 G ops/sec  
CPU: 8192 x 8192 matmul took: 39.51 sec, 27.83 G ops/sec   

### 2) VGG 16

#### A) Keras  
Python script : keras_VGG16

100 iterations  
batch_size = 16 
x = 224x224x3, zeros
y = 1000, zeros

Machine 1  
GPU: Time per iteration: 158.891 ms  

Machine 2  
GPU: 


