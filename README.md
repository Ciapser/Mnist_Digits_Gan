# Mnist Digits Gan
It's nothing more, and nothing less. This repository contains generative adversarial network trained on Mnist Digits dataset.

Script was written using NeuroUtils Library which is based mainly on Tensorflow and Numpy libraries
## Model
To define architectures you should download Tensorflow library

	import tensorflow as tf
	
### Architecture used in the generator:

	def Generator(latent_dim):
	    inputs = tf.keras.layers.Input((latent_dim))
	     
	     # foundation for 7x7 image
	     n_nodes = 128 * 7 * 7
	     x = tf.keras.layers.Dense(n_nodes)(inputs)
	     x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
	     x = tf.keras.layers.Reshape((7, 7, 128))(x)
	     
	     # upsample to 14x14
	     x = tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same')(x)
	     x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
	     
	     x = tf.keras.layers.Conv2D(256, (4,4), strides=(1,1), padding='same')(x)
	     x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
	     
	     # upsample to 28x28
	     x = tf.keras.layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same')(x)
	     x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
	     
	     x = tf.keras.layers.Conv2D(128, (4,4), strides=(1,1), padding='same')(x)
	     x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
	     
	     outputs = tf.keras.layers.Conv2D(1, (7,7), activation='tanh', padding='same')(x)

	     model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
	     
	     return model


### Architecture used in the discriminator:

    def Discriminator(in_shape=(28,28,1)):
        inputs = tf.keras.layers.Input(in_shape)
        
        #Downsampling to 14x14
        x = tf.keras.layers.Conv2D(128, (3,3), strides=(2, 2), padding='same')(inputs)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        #Convolution layer
        x = tf.keras.layers.Conv2D(128, (3,3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        #Downsample to 7x7
        x = tf.keras.layers.Conv2D(256, (3,3), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        #Convolution layer
        x = tf.keras.layers.Conv2D(256, (3,3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        #Flattening output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        
        return model




## Results
### Generator outputs:
![alt text](https://github.com/Ciapser/Mnist_Digits_Gan/blob/master/Actual_Best_Result/Generator_results.png?raw=true)

### Generator training:
![alt text](https://github.com/Ciapser/Mnist_Digits_Gan/blob/master/Actual_Best_Result/Training_history.gif?raw=true)

### Outputs interpolation

![alt text](https://github.com/Ciapser/Mnist_Digits_Gan/blob/master/Actual_Best_Result/Model_interpolation.gif?raw=true)


