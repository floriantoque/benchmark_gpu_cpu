## Benchmark GPU vs CPU (Python 3)

Machine 1  
  * GPU: GEFORCE GTX 1080 Ti (12GB)    
  * CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz  
  * Python 3.5, Keras : 2.2.2, Tensorflow version : 1.9.0, Numpy version: 1.13.1  
          
Machine 2  
  * GPU: Tesla V100 (32GB)    
  * CPU: Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz  
  * env Tensorflow : Python 3.6.5, Tensorflow version : 1.8.0, Numpy version: 1.14.3
  * env Keras : Python 3.6.5, Keras : 2.2.2, Tensorflow version : 1.9.0


### 1) Matrix multiplication
Python script : matmul.py

10 iterations  
Matrix1 : shape(8192,8192), values 1  
Matrix2 : shape(8192,8192), values 1  

Machine 1  
GPU: (1 GPU) 8192 x 8192 matmul took: 0.23 sec, 4805.72 G ops/sec  
CPU: 8192 x 8192 matmul took: 5.75 sec, 191.34 G ops/sec  

Machine 2  (env Tensorflow)
GPU: (1 GPU) 8192 x 8192 matmul took: 0.08 sec, 13581.87 G ops/sec  
CPU: 8192 x 8192 matmul took: 39.51 sec, 27.83 G ops/sec   

### 2) VGG 16

#### A) Keras  (batch_size 16, few operations per iteration)
Python script : keras_VGG16  

100 iterations    
batch_size = 16    
x = 224x224x3, zeros        
y = 1000, zeros        

Machine 1   
GPU: (1 GPU) Time per iteration: 158.891 ms    

Machine 2  (env Keras)   
GPU: (1 GPU) Time per iteration:  95.611 ms  
GPU: (2 GPU) Time per iteration: 286.549 ms  
GPU: (3 GPU) Time per iteration: 357.761 ms    
 
#### B) Keras  (batch_size 256, lot of operations per iteration)
Python script : keras_VGG16

10 iterations   
batch_size = 256  
x = 224x224x3, zeros  
y = 1000, zeros  

Machine 1
GPU: (1 GPU) ResourceExhaustedError

Machine 2  (env Keras)
GPU: (1 GPU) Time per iteration: 1332.705 ms   
GPU: (2 GPU) Time per iteration: 831.424 ms   
GPU: (3 GPU) Time per iteration: 823.483 ms 
