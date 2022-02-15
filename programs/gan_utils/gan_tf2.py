import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import time 

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

    @tf.function
    def g_train_step(self, x_real):

        with tf.GradientTape() as gen_tape:
            z = self.random_noise()

            x_fake = self.g_net(z, training=True)

            y_hat_fake = self.d_net(x_fake, training=False)

            g_loss = tf.reduce_mean(y_hat_fake)

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

            d_loss = tf.reduce_mean(y_hat_fake) - tf.reduce_mean(y_hat_real) 

        d_grads = disc_tape.gradient(d_loss, self.d_net.trainable_variables)

        self.d_optimizer.apply_gradients(zip(d_grads, self.d_net.trainable_variables))

        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_net.trainable_variables]

        return d_loss

    def train(self, dataset, nepochs, batch_size, output_examples):
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