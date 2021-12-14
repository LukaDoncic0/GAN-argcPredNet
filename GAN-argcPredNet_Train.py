from keras.layers import Input, Dense,  Flatten, Lambda, Concatenate, LeakyReLU, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from functools import partial
from function import RandomWeightedAverage, wasserstein_loss, gradient_penalty_loss
from function import simple_slice, subimg
from Argc_PredNet import Argc_PredNet
import numpy as np
import hickle as hkl
import os
import tensorflow as tf
from function import compute_train_ssim, saveimages, compute_ssim_mul
from skimage.measure import compare_mse as mse
from tensorflow.compat.v1 import ConfigProto,InteractiveSession
import keras.backend.tensorflow_backend as KTF

class WGANGP():
    def __init__(self, img_rows, img_cols,):
        # Training parameters
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = 1
        self.sequences = 24
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = (5,) + self.img_shape
        self.d_input_shape = (self.img_rows, self.img_cols, 2)
        self.input_img_num = 5
        self.index = 5
        self.n_critic = 5
        adam_opt = Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.d_input_shape, name='real_img')

        # Noise input
        z_disc = Input(shape=self.latent_dim, name='z_disc')
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)
        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critical_model = Model(inputs=[real_img, z_disc],
                             outputs=[valid, fake, validity_interpolated])
        self.critical_model.compile(loss=[wasserstein_loss,
                                   wasserstein_loss,
                                   partial_gp_loss],
                             optimizer=adam_opt,
                             loss_weights=[1, 1, 10])
        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=self.latent_dim, name='z_gen')
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=wasserstein_loss, optimizer=adam_opt)

        # argcPredNet Generator
    def build_generator(self):
        stack_sizes = (1, 128, 128, 256)
        R_stack_sizes = stack_sizes
        A_filt_sizes = (3, 3, 3)
        Ahat_filt_sizes = (3, 3, 3, 3)
        R_filt_sizes = (3, 3, 3, 3)
        prednet = Argc_PredNet(stack_sizes, R_stack_sizes,
                               A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                               output_mode='prediction', return_sequences=True)

        noise = Input(self.latent_dim, name='g_model_input')
        prediction = prednet(noise)
        result_img = Lambda(simple_slice, output_shape=self.img_shape, arguments={'index': self.index})(prediction)
        real_img = Lambda(simple_slice, output_shape=self.img_shape, arguments={'index': self.index})(noise)
        generated_images = Concatenate(axis=3)([real_img, result_img])
        return Model(noise, generated_images)
    
    #dual-channel Discriminator
    def build_critic(self):
        input = Input(shape=self.d_input_shape, name='d_model_input')
        mix_con2d_layer1 = Conv2D(32, kernel_size=3, strides=2, input_shape= self.d_input_shape, padding="same")(input)
        mix_lre_layer1 = LeakyReLU(alpha=0.2)(mix_con2d_layer1)

        mix_con2d_layer2 = Conv2D(64, kernel_size=3, strides=2, padding="same")(mix_lre_layer1)
        mix_lre_layer2 = LeakyReLU(alpha=0.2)(mix_con2d_layer2)
        
        mix_con2d_layer3 = Conv2D(128, kernel_size=3, strides=2, padding="same")(mix_lre_layer2)
        mix_lre_layer3 = LeakyReLU(alpha=0.2)(mix_con2d_layer3)
        
        mix_con2d_layer4 = Conv2D(256, kernel_size=3, strides=2, padding="same")(mix_lre_layer3)
        mix_lre_layer4 = LeakyReLU(alpha=0.2)(mix_con2d_layer4)

        mix_flatten_layer = Flatten()(mix_lre_layer4)

        mix_dense_layer1 = Dense(1024)(mix_flatten_layer)
        mix_dense_layer2 = Dense(1)(mix_dense_layer1)

        return Model(input, mix_dense_layer2)

    def train(self, BATCH_SIZE, epochs, train_file, val_file, picsave_dir, weights_dir):

        # Rescale 0~1
        X_train = hkl.load(train_file) / 255
        X_train = np.expand_dims(X_train, axis=4)

        X_val = hkl.load(val_file) / 255
        X_val = np.expand_dims(X_val, axis=4)

        # Adversarial ground truths
        valid = -np.ones((BATCH_SIZE, 1))
        fake = np.ones((BATCH_SIZE, 1))
        dummy = np.zeros((BATCH_SIZE, 1))  # Dummy gt for gradient penalty
        for epoch in range(epochs):
            for frame in range(20):
                for _ in range(self.n_critic):
                    #----------------------
                    # Train Discriminator
                    #----------------------
                    idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)  # generate a random batch_size length of list from 0-4807
                    imgs = X_train[idx]
                    noise = imgs[:, frame: frame + self.input_img_num + 1]
                    image_batch = imgs[:, frame + self.input_img_num]
                    image_batch = np.concatenate([image_batch, image_batch], axis=3)
                    # Train the critic
                    d_loss = self.critical_model.train_on_batch([image_batch, noise], [valid, fake, dummy])
                #------------------
                # Train Generator
                #------------------
                g_loss = self.generator_model.train_on_batch(noise, valid)

                if frame % 10 == 0 and epoch % 5 == 0:
                    compute_train_ssim(self.generator, X_val, epoch, frame, d_loss[0], -g_loss,
                                       4, X_train.shape[1] - 4, self.img_rows, self.img_cols,
                                       picsave_dir)
            if epoch % 5 == 0:
                self.generator.save_weights(weights_dir+'gen_weight'+epoch + '.h5')
                self.critic.save_weights(weights_dir+'dis_weight' + epoch + '.h5')

# Training parameters
img_rows = 96
img_cols = 96

# Data files
DATA_DIR = '/run/media/root/8TB/png[0]/layer[0]/'
train_file = os.path.join(DATA_DIR, 'train.hkl')        # shape = (sequence_num,frames,img_rows, img_cols, channels)  The number of frames in each batch is 12
val_file = os.path.join(DATA_DIR, 'val.hkl')            # shape = (one_sequence,frames,img_rows, img_cols, channels)

Pic_dir = '/run/media/root/8TB/png[0]/train_result/train_image/'
Weights_Dir = '/run/media/root/8TB/png[0]/train_result/weights/'

# GAN-argcPredNet train
wgan = WGANGP(img_rows, img_cols, )
wgan.train(BATCH_SIZE=4, epochs=8, train_file=train_file, val_file=val_file, picsave_dir=Pic_dir, weights_dir=Weights_Dir, )
