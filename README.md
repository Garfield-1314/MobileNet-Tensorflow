# Tflite
# Tensorflow Lite
 
 # Tensorflow Lite 模型训练并进行全整型量化脚本，运行tflite.ipynb中的代码训练模型
 # 网络为MobileNetV1，可替换为V2等

 # 训练环境: Python3.8.20 ,Cuda 11.8 ,Tensorflow-gpu 2.10.0

# 10_keyboard,11_mobile_phone,12_mouse,13_headphones,14_monitor,15_speaker,1_wrench,2_soldering_iron,3_electrodrill,4_tape_measure,5_screwdriver,6_pliers,7_oscillograph,8_multimeter,9_printer

# 以下是安装顺序（）
# 创建的python环境为3.8.20

# 1、安装windows下的cuda11.8和cudnn
# 2、安装conda环境下的tensorflow：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu==2.10.0
# 3、安装conda环境下的matplotlib、seaborn、scikit-learn：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib seaborn scikit-learn opencv-Python tqdm tensorflow_model_optimization