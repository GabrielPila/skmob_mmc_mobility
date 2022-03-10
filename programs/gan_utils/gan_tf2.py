import json
import os
import time 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

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

    def create_generator(self):
        G = tf.keras.Sequential()
        G.add(tf.keras.Input(shape=(self.random_dim,)))
        for u in self.generatorDims[:-1]:
            G.add(tf.keras.layers.Dense(units=u))
            G.add(tf.keras.layers.BatchNormalization())
            G.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        G.add(tf.keras.layers.Dense(self.generatorDims[-1], activation="relu"))

        return G

    def create_discriminator(self):
        D = tf.keras.Sequential()
        D.add(tf.keras.Input(shape=(self.input_dim,)))
        for u in self.discriminatorDims[:-1]:
            D.add(tf.keras.layers.Dense(units=u))
            D.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        D.add(tf.keras.layers.Dense(self.discriminatorDims[-1], activation="relu"))

        return D

    def random_noise(self, nrows=None):
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

    def g_apply_loss_fun( y_pred, loss_function=None):

        ##If str: pick from list and return the corresponding function
        if (isinstance(loss_function, str)):
            return getattr(tf, loss_function)(y_pred)

        ## If function, return it
        elif (callable(loss_function)):
            return loss_function(y_pred)
            
        ## Default: 
        else:
            mse = tf.keras.losses.MeanSquaredError()
            return -mse(y_pred).numpy()
         
    def d_apply_loss_fun(y_pred, y_real, loss_function=None ):

        ##If str: pick from list and return the corresponding function
        if (isinstance(loss_function, str)):
            return getattr(tf, loss_function)(y_pred) - getattr(tf, loss_function)(y_real)

        ## If function, return it
        elif (callable(loss_function)):
            return loss_function(y_pred) - loss_function(y_real)
            
        ## Default: 
        else:
            mse = tf.keras.losses.MeanSquaredError()
            return mse(y_pred).numpy() -mse(y_real).numpy()         

    @tf.function
    def g_train_step(self, x_real, g_loss_function):


        with tf.GradientTape() as gen_tape:
            z = self.random_noise()

            x_fake = self.g_net(z, training=True)

            y_hat_fake = self.d_net(x_fake, training=False)

            g_loss = self.g_apply_loss_fun(g_loss_function, y_hat_fake) 

        g_grads = gen_tape.gradient(g_loss, self.g_net.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_grads, self.g_net.trainable_variables))

        return g_loss

    @tf.function
    def d_train_step(self, x_real):

        with tf.GradientTape() as disc_tape:
            z = self.random_noise()

            x_fake = self.g_net(z, training=False)

            y_hat_real = self.d_net(x_real, training=True)
            y_hat_fake = self.d_net(x_fake, training=True)

            d_loss = self.d_apply_loss_fun(y_hat_fake, y_hat_real, loss_function=None ) 

        d_grads = disc_tape.gradient(d_loss, self.d_net.trainable_variables)

        self.d_optimizer.apply_gradients(zip(d_grads, self.d_net.trainable_variables))

        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_net.trainable_variables]

        return d_loss

    def train(self, dataset, nepochs, batch_size, output_examples, g_loss_function=None, d_loss_function = None):
        self.batch_size = batch_size
        self.output_examples = output_examples
        with tf.device("gpu:0"):
            for epoch in range(nepochs):
                start_time = time.time()
                pbar = tqdm(dataset, disable=True)
                pbar.set_description("Epoch {}/{}".format(epoch+1, nepochs))
                for batch in pbar:
                    batch = tf.cast(batch, tf.float32)
                    rd_loss = self.d_train_step(batch)
                    rg_loss = self.g_train_step(batch)
                    
                    self.g_loss_store.append(rg_loss.numpy())
                    self.d_loss_store.append(rd_loss.numpy())

        z_sample = self.random_noise(self.output_examples)
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
        fig = plt.figure(figsize=(12,4))

        fig.add_subplot(1, 2, 1)
        plt.plot(self.g_loss_store)
        plt.title('Generator Losses')
        plt.xlabel('step')

        fig.add_subplot(1, 2, 2)
        plt.plot(self.d_loss_store)
        plt.title('Discriminator Losses')
        plt.xlabel('step')

        plt.tight_layout();
        plt.savefig(os.path.join(path, 'losses.jpg'));