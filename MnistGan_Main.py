import Config
from NeuroUtils import Core

import numpy as np
import os
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from tqdm import tqdm
import random
import sys
from contextlib import redirect_stdout

#1
#Creating Class of the project, putting parameters from Config file
Mnist = Core.Project.Classification_Project(Config)

#2
#Initializating data from main database folder to project folder. 
#Parameters of this data like resolution and crop ratio are set in Config
Mnist.Initialize_data()
####################################################



####################################################
#3
#Loading and merging data to trainable dataset.
x1 = np.load(os.path.join(Mnist.DATA_DIRECTORY , "x_train.npy"))
x2 = np.load(os.path.join(Mnist.DATA_DIRECTORY , "x_test.npy"))

train_images = np.vstack((x1,x2))
train_images = (train_images-0.5)*2

os.makedirs('Images', exist_ok=True)
# Hyperparameters
latent_dim = 100
batch_size = 256
epochs = 100
sample_interval = 5

# Build the generator
def build_generator(latent_dim):
    model = Sequential()
    # foundation for 7x7 image
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    # upsample to 14x14
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(128, (4,4), strides=(1,1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 28x28
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(128, (4,4), strides=(1,1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(1, (7,7), activation='tanh', padding='same'))
    return model


# Build the discriminator
def build_discriminator(in_shape=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(64, (3,3), strides=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(64, (3,3), strides=(1, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(Dense(1, activation='sigmoid'))
    
    return model
  
  
#Build combined GAN network
def build_gan(gan_generator ,gan_discriminator):
    #1
    #Freeze the discriminator's weights
    gan_discriminator.trainable = False  
    
    gan_input = tf.keras.Input(shape=(latent_dim,))
    data = generator(gan_input)
    gan_output = discriminator(data)
    gan = tf.keras.Model(gan_input, gan_output)
    return gan


#Generating real samples from dataset
def generate_real_samples(dataset, n_samples):
    #Generate random indexes
    idx = np.random.randint(0, len(dataset), n_samples)
    #Get random images from dataset
    x = dataset[idx]
    #generating labels for real class
    y = np.ones((n_samples, 1))
    return x, y


#Generating fake samples using noise and generator
def generate_fake_samples(gan_generator, latent_dim, n_samples):
    #generate noise as input for generator
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    # predict outputs
    x = gan_generator.predict(noise)
    if len(x.shape) == 4:
        x = np.squeeze(x, axis = -1)
    # create 'fake' class labels (0)
    y = np.zeros((n_samples, 1))
    return x, y


#Saving progress each n epochs to folder
def save_plot(examples,directory, epoch, n=10):
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        # plot raw pixel data
        plt.imshow(examples[i, :, :])
        # save plot to file
    filename = os.path.join(directory, 'Epoch: %03d' %epoch )

    plt.savefig(filename)
    plt.close()
    
    
# Training the GAN Network
def train_gan(dataset, gan_model, gan_generator, gan_discriminator, epochs, batch_size, sample_interval):

    for epoch in range(epochs):
        print("\nEpoch:",epoch)
        steps_per_epoch = len(dataset) // batch_size
        for step in tqdm(range(steps_per_epoch)):
            with redirect_stdout(open(os.devnull, 'w')):
                #1
                #Taking batch of real samples from dataset
                x_real, y_real = generate_real_samples(dataset, batch_size//2)
                
                #2
                #Generating batch of fake samples from generator
                x_fake , y_fake = generate_fake_samples(gan_generator, latent_dim, batch_size//2)
                
                #3
                #Preparing combined real-fake set for discriminator to train
                x = np.vstack((x_real,x_fake))
                y = np.vstack((y_real, y_fake))
                
                #4
                #Training discriminator
                discriminator_loss = gan_discriminator.train_on_batch(x,y)
                
                #5
                #Update generator via discriminator error
                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                ones = np.ones((batch_size, 1))
                generator_loss = gan_model.train_on_batch(noise, ones)
                
        # Print the progress
        sys.stdout.write(f"[D loss: {discriminator_loss[0]:.3f} | D acc: {discriminator_loss[1]:.3f}] [G loss: {generator_loss:.3f}]")    
       
        # Save generated images every sample_interval
        if epoch % sample_interval == 0:
            save_plot(x_fake,"Images",epoch,8)    






discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

generator = build_generator(latent_dim)

Gan = build_gan(generator, discriminator)
Gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))






# Train the GAN
with tf.device("GPU:0"):
    train_gan(dataset = train_images,
              gan_model = Gan,
              gan_generator = generator,
              gan_discriminator = discriminator,
              epochs = epochs,
              batch_size = batch_size,
              sample_interval = sample_interval)


