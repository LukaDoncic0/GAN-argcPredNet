from keras.layers import Input, Dense,  Flatten, Lambda, Concatenate, LeakyReLU, Conv2D
from keras.models import  Model
from keras.optimizers import Adam
from functools import partial
from function import RandomWeightedAverage, wasserstein_loss, gradient_penalty_loss, extrap_loss
from function import simple_slice
from Argc_PredNet import Argc_PredNet

# Model parameters
stack_sizes = (1, 128, 128, 256)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)

# argcPredNet Generator
def build_Argc_PredNet_model(latent_dim, img_shape, input_img_num):
    prednet = Argc_PredNet(stack_sizes, R_stack_sizes,
                      A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                      output_mode='prediction', return_sequences=True)
    noise = Input(latent_dim, name='g_model_input')
    prediction = prednet(noise)
    result_img = Lambda(simple_slice, output_shape=img_shape, arguments={'index': input_img_num})(prediction)
    real_img = Lambda(simple_slice, output_shape=img_shape, arguments={'index': input_img_num})(noise)
    generated_images = Concatenate(axis=3)([real_img, result_img])
    return Model(noise, generated_images)
def build_generator(latent_dim, img_shape,input_img_num):
    g = build_Argc_PredNet_model(latent_dim, img_shape, input_img_num)
    return g

#dual-channel Discriminator
def d_model_1(input, d_input_shape):
    mix_con2d_layer1 = Conv2D(128, kernel_size=3, strides=2, input_shape=d_input_shape, padding="same")(input)
    mix_lre_layer1 = LeakyReLU(alpha=0.2)(mix_con2d_layer1)
    mix_con2d_layer2 = Conv2D(64, kernel_size=3, strides=2, padding="same")(mix_lre_layer1)
    mix_lre_layer2 = LeakyReLU(alpha=0.2)(mix_con2d_layer2)
    mix_flatten_layer = Flatten()(mix_lre_layer2)

    mix_dense_layer1 = Dense(1024)(mix_flatten_layer)
    mix_dense_layer2 = Dense(1)(mix_dense_layer1)
    return Model(input, mix_dense_layer2)

def build_discriminator_model(d_input_shape):
    img = Input(shape=d_input_shape, name='d_model_input')
    model1 = d_model_1(img, d_input_shape)
    out1 = model1.output
    return Model(img, out1)

# Build WGan-gp
def build_gan(BATCH_SIZE, d_input_shape, latent_dim, img_shape, img_rows, img_cols, input_img_num):
    g = build_generator(latent_dim, img_shape, img_rows, img_cols, input_img_num)
    d = build_discriminator_model(d_input_shape)
    # optimizer = RMSprop(lr=0.00005)
    adam_opt = Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)
    # -------------------------------
    # Construct Computational Graph
    #       for the Critic
    # -------------------------------

    # Freeze generator's layers while training critic
    g.trainable = False

    real_img = Input(shape=d_input_shape, name='real_img')
    z_disc = Input(shape=latent_dim, name='z_disc')

    fake_img = g(z_disc)
    fake = d(fake_img)
    valid = d(real_img)
    interpolated_img = RandomWeightedAverage()([real_img, fake_img])
    validity_interpolated = d(interpolated_img)
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty'

    critic_model = Model(inputs=[real_img, z_disc],
                   outputs=[valid, fake, validity_interpolated])
    critic_model.compile(loss=[wasserstein_loss,
                         wasserstein_loss,
                         partial_gp_loss],
                   optimizer=adam_opt,
                   loss_weights=[1, 1, 10])
    # -------------------------------
    # Construct Computational Graph
    #         for Generator
    # -------------------------------

    # For the generator we freeze the critic's layers
    d.trainable = False
    g.trainable = True

    # Sampled noise for input to generator
    z_gen = Input(shape=latent_dim, name='z_gen')
    # Generate images based of noise
    img = g(z_gen)
    # Discriminator determines validity
    valid = d(img)
    # Defines generator model
    generator_model = Model(z_gen, valid)
    generator_model.compile(loss=wasserstein_loss, optimizer=adam_opt)
    return d, g, critic_model, generator_model


#Build argcPredNet
def build_argcPredNet(nt, img_rows, img_cols):
    input_shape = (img_rows, img_cols, 1)
    prednet = Argc_PredNet(stack_sizes, R_stack_sizes,
                      A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                      output_mode='prediction', return_sequences=True)
    inputs = Input(shape=(nt,) + input_shape)
    prediction = prednet(inputs)
    model = Model(input=inputs, output=prediction)
    model.compile(loss=extrap_loss, optimizer='adam')
    return model


