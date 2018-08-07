##Benchmark GPU vs CPU (Python 3)

Machine 1  
  * GPU: GEFORCE GTX 1080 Ti (12G)  
  * CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz  
  * Python 3.5, Tensorflow version : 1.9.0, Numpy version: 1.13.1  
          
Machine 2  
  * GPU:  
  * CPU:   



### 1) Matrix multiplication
Python script : matmul.py

10 iterations  
Matrix1 : shape(8192,8192), values 1  
Matrix2 : shape(8192,8192), values 1  

Machine 1  
GPU: 8192 x 8192 matmul took: 0.23 sec, 4805.72 G ops/sec  
CPU: 8192 x 8192 matmul took: 5.75 sec, 191.34 G ops/sec




