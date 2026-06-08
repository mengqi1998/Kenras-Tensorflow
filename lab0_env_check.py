import sys
import os
import sklearn
import tensorflow
import numpy
import keras

print("python的intepreter在:{}".format(sys.executable))
print("目前的工作目錄:{}".format(os.getcwd()))
print("sklearn的版本:{}".format(sklearn.__version__))
print("tensorflow version:{}".format(tensorflow.__version__))
print("numpy version:{}".format(numpy.__version__))
print("keras version:{}".format(keras.__version__))