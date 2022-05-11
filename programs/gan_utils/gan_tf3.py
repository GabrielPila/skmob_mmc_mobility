import json
import os
import time 

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import pylab as pl
from IPython import display
from tqdm import tqdm
tf.config.run_functions_eagerly(True)

class GAN:

    def __init__(
        self,
        input_dim,
        random_dim,
        discriminatorDims,
        generatorDims,
        g_optimizer,
        d_optimizer
    ):

        self.input_dim = input_dim
        self.random_dim = random_dim
        self.discriminatorDims = discriminatorDims
        self.generatorDims = generatorDims

        self.g_net = self.create_generator()
        self.d_net = self.create_discriminator()

        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.g_loss_store = []
        self.d_loss_store = []
        self.wdis_store   = []
        self.data_limits = None

    def create_generator(self):
        input_shape = self.random_dim
        inputs = tf.keras.layers.Input(input_shape)
        net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name='fc1')(inputs)
        net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name='fc2')(net)
        net = tf.keras.layers.Dense(units=self.input_dim, name='G')(net)
        G = tf.keras.Model(inputs=inputs, outputs=net)
        print('generator created...')

        return G

    def create_discriminator(self):
        input_shape = self.input_dim+3
        inputs = tf.keras.layers.Input(input_shape)
        net = tf.keras.layers.Dense(units=32, activation=tf.nn.elu, name='fc1')(inputs)
        net = tf.keras.layers.Dense(units=1, name='D')(net)
        D = tf.keras.Model(inputs=inputs, outputs=net)
        print('discriminator created...')
        return D 

    def random_noise(self, nrows=None):
        if self.data_limits != None:

            results = tf.reshape(tf.constant([], dtype=tf.float32),(self.batch_size if not nrows else nrows,0))
            for col in self.data_limits:
                results = tf.concat([results,
                    tf.random.uniform(
                        (self.batch_size if not nrows else nrows,1),
                        minval=self.data_limits[col]['min'],
                        maxval=self.data_limits[col]['max']
                        )],axis=1)

            return results
        else:
            return tf.random.uniform(
                (self.batch_size if not nrows else nrows, self.random_dim),
                minval=-1, maxval=1
            )

    def dpnoise(self, tensor):
        '''add noise to tensor'''
        s = tensor.get_shape().as_list()  # get shape of the tensor

        rt = tf.random.normal(s, mean=0.0, stddev=self.noise_std)
        t = tf.add(tensor, tf.scalar_mul((1.0 / self.batch_size), rt))
        return t
    

    def d_loss(self,real_output, generated_output):
        '''Discriminator Loss Function'''
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        d_loss = bce(tf.ones_like(real_output), real_output)\
                + bce(tf.zeros_like(generated_output), generated_output)
        return d_loss

    def g_loss(self,generated_output):
        '''Generator Loss Function'''
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        g_loss = bce(tf.ones_like(generated_output), generated_output)
        return g_loss        

    def train(self, dataset, nepochs, batch_size,):
        # Define the optimizers and the train operations
        optimizer = tf.keras.optimizers.Adam(1e-3)

        self.nepochs = nepochs
        self.epoch_size = dataset.shape[0]/ batch_size

        @tf.function
        def train_step():
            with tf.GradientTape(persistent=True) as tape:
                real_data = dataset
                noise_vector = tf.random.normal(
                    mean=80, stddev=10, shape=(real_data.shape[0], self.random_dim)
                )
                # Sample from the Generator
                fake_data = self.g_net(noise_vector, training=False)

                # Compute norm of fake_data
                fake_data = tf.concat([fake_data,\
                    tf.math.square(fake_data[:,1:] ),\
                    tf.reduce_sum(tf.math.square(fake_data[:,1:] ), axis=1, keepdims=True)],\
                    axis=1) 
                
                

                # Compute the D loss
                d_fake_data = self.d_net(fake_data)
                d_real_data = self.d_net.predict(real_data)
                d_loss_value = self.d_loss(generated_output=d_fake_data, real_output=d_real_data)
                # Compute the G loss
                g_loss_value = self.g_loss(generated_output=d_fake_data)


            self.g_loss_store.append(g_loss_value.numpy())
            self.d_loss_store.append(d_loss_value.numpy())
            # Now that we have computed the losses, we can compute the gradients 
            # (using the tape) and optimize the networks
            d_gradients = tape.gradient(d_loss_value, self.d_net.trainable_variables)
            g_gradients = tape.gradient(g_loss_value, self.g_net.trainable_variables)
            del tape

            # Apply gradients to variables
            optimizer.apply_gradients(zip(d_gradients, self.d_net.trainable_variables))
            optimizer.apply_gradients(zip(g_gradients, self.g_net.trainable_variables))
            return real_data, fake_data, g_loss_value, d_loss_value

        # 40000 training steps with logging every 200 steps
        
        
        for step in range(nepochs):
            real_data, fake_data, g_loss_value, d_loss_value = train_step()
            if step % 100 == 0:
                print(
                    "G loss: ",
                    g_loss_value.numpy(),
                    " D loss: ",
                    d_loss_value.numpy(),
                    " step: ",
                    step,
                )
                fig, ax = plt.subplots(nrows = 2, ncols=2, figsize=(10,10))
                plt.title('Data Real vs Data Generated')
                # Sample 5000 values from the Generator and draw the histogram
                total_real_data = np.concatenate([real_data[:,1].numpy(),real_data[:,2].numpy()])
                total_fake_data = np.concatenate([fake_data[:,1].numpy(),fake_data[:,2].numpy()])
                print(total_real_data)
                print(total_fake_data)

                ## Plot latitude
                ax[0][0].hist(real_data[:,1].numpy(), label = 'real', alpha = 0.5)
                ax[0][0].hist(fake_data[:,1].numpy(), label = 'fake', alpha = 0.5) 
                ax[0][0].set_title('Latitude')

                ## Plot longitude
                ax[0][1].hist(real_data[:,2].numpy(), label = 'real', alpha = 0.5)
                ax[0][1].hist(fake_data[:,2].numpy(), label = 'fake', alpha = 0.5)
                ax[0][1].set_title('Longitude')

                ## Plot Hour
                ax[1][0].hist(real_data[:,0].numpy(), label = 'real', alpha = 0.5)
                ax[1][0].hist(fake_data[:,0].numpy(), label = 'fake', alpha = 0.5)
                ax[1][0].set_title('Hour')

                ## Scatter of Lat x Lon
                ax[1][1].scatter(x = real_data[:,1].numpy(), y = real_data[:,2].numpy(), label = 'real', alpha = 0.5)
                ax[1][1].scatter(x = fake_data[:,1].numpy(), y = fake_data[:,2].numpy(), label = 'fake', alpha = 0.5)
                ax[1][1].set_title('Latitude x Longitude')

                # these are matplotlib.patch.Patch properties
                props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

                # place a text box in upper left in axes coords
                
                textstr = f"step={step}"
                ax[0][0].text(
                    0.05,
                    0.95,
                    textstr,
                    transform=ax[0][0].transAxes,
                    fontsize=14,
                    verticalalignment="top",
                    bbox=props,
                )

                #axes = plt.gca()
                #display.display(plt.gcf())
                #display.clear_output(wait=True)
                #plt.savefig("./gif/{}.png".format(step))
                plt.legend()
                plt.show()
                #plt.gca().clear()
                
                self.plot_loss_progress()

        z_sample = tf.random.normal(
                    mean=0, stddev=1, shape=(real_data.shape[0], self.random_dim)
                )
        x_gene = self.g_net.predict(z_sample)
        return x_gene




    def generate_data(self, num_samples):
        z_sample = self.random_noise(num_samples)
        x_gene = self.g_net.predict(z_sample)
        return x_gene

    
    
    def save_models(self, path='./'):
        path_models = os.path.join(path, 'models')
        if not os.path.exists(path_models):
            os.mkdir(path_models)
        self.d_net.save(os.path.join(path_models, 'dnet.h5'))
        self.g_net.save(os.path.join(path_models, 'gnet.h5'))
        
        try:
            d_loss = [float(x) for x in self.d_loss_store]
            g_loss = [float(x) for x in self.g_loss_store]
            json.dump(d_loss, open(os.path.join(path_models, 'd_loss.json'), 'w'))
            json.dump(g_loss, open(os.path.join(path_models, 'g_loss.json'), 'w'))
        except:
            print('Losses could not be saved')
            pass
        
        print(f'Models saved in {path_models}!')
        
    def load_models(self, path='./'):
        path_models = os.path.join(path, 'models')
        if os.path.exists(path_models):
            self.d_net = tf.keras.models.load_model(os.path.join(path_models, 'dnet.h5'))
            self.g_net = tf.keras.models.load_model(os.path.join(path_models, 'gnet.h5'))
            
            try:
                self.d_loss_store = json.load(open(os.path.join(path_models, 'd_loss.json'), 'r'))
                self.g_loss_store = json.load(open(os.path.join(path_models, 'g_loss.json'), 'r'))            
            except:
                print('Losses could not be loaded')
                pass                
            print(f'Models loaded from {path_models}!')
        else:
            print(f'Path "{path_models}" not found.')
            
    
    def plot_loss_progress(self, path='./'):
        import math
        def custom_format_epoch_func(value,tick_number):
            return 'Epoch {:,.0f}'.format(value//self.epoch_size)

        fig,ax = plt.subplots(2,1,figsize=(12,4))
        plt.title('Losses over epochs')
        # Generator Losses
        ax[0].plot(self.g_loss_store)
        ax[0].set_xlabel('step')
        ax[0].set_title('Generator Losses')
        #ax[0].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
        #ax[0].xaxis.set_major_locator(matplotlib.ticker.FixedLocator([math.ceil(i*self.epoch_size) for i in range(0,self.nepochs)]))
        #ax[0].xaxis.set_major_formatter(plt.FuncFormatter(custom_format_epoch_func))
        #ax[0].xaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        #ax[0].tick_params(axis='both', which='minor', labelsize=8)
        #ax[0].tick_params(axis='x', which='major', width=2, labelcolor ='r')

        # Discriminator Losses
        ax[1].plot(self.d_loss_store)
        ax[1].set_xlabel('step')
        ax[1].set_title('Discriminator Losses')

        plt.subplots_adjust(hspace=0.8)
        #ax[1].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
        #ax[1].xaxis.set_major_locator(matplotlib.ticker.FixedLocator([math.ceil(i*self.epoch_size) for i in range(0,self.nepochs)]))
        #ax[1].xaxis.set_major_formatter(plt.FuncFormatter(custom_format_epoch_func))
        #ax[1].xaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        #ax[1].tick_params(axis='both', which='minor', labelsize=8)
        #ax[1].tick_params(axis='x', which='major', width=2, labelcolor ='r')
        plt.show()

        plt.tight_layout();
        #plt.savefig(os.path.join(path, 'losses.jpg'));

    