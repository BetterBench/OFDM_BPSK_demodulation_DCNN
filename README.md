# 1 内容

利用Deeplearning 深度一维卷积网络实现OFDM系统中BPSK的信号解调，把复数信号拆分为实部和虚部合并成一维数据进行cov1D卷积,实现信号的解调

# 2 数据集
数据集来自OFDM发送机[https://github.com/823316627bandeng/OFDM_Rician_simulation_for_DeepLearning](https://github.com/823316627bandeng/OFDM_Rician_simulation_for_DeepLearning)
BPSK的调制方式，数据帧的数量可以自定义，该目录下的bpsk-data100.mat只是一个小数据量，仅作为测试程序用，没有训练价值。

# 3 模型
该模型实现的对复数信号进行处理，拉直为1维的数据，进行卷积的的卷积神经网络

-- TrainModel32.py 数据集精度float32,对应是使用的模型是Model32.py,还在调试该程序，未测试通

-- TrainModel64.py 数据集精度float64,对应是使用的模型是Model64.py，已经测试通，在1万帧训练集下，准确率达51.2%

# 4 环境
TensorFlow-gpu 1.15
python 3.6

# 5 运行
> python TrainModel64.py
