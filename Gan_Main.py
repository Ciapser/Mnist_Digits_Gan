import sys
import Config
sys.path.insert(0, 'C:\\Users\Stacja Robocza\\Desktop\\NeuroUtils\\Tests')
import Core

import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from contextlib import redirect_stdout
import os

#1
#Creating Class of the project, putting parameters from Config file
Gan = Core.Project.Gan_Project(Config)

#2
#Initializating data from main database folder to project folder. 
#Parameters of this data like resolution and crop ratio are set in Config
Gan.Initialize_data()

Gan.Load_and_merge_data()

Gan.Process_data()


Gan.Initialize_model_from_library()

x = Gan.X_TRAIN
#plt.imshow(x[1])
#plt.figure()
#plt.imshow((x[1]+1)/2)


g = Gan.GENERATOR
d = Gan.DISCRIMINATOR
m = Gan.MODEL

#####################################################################




Gan.Initialize_weights_and_training_gan()





