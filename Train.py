import numpy as np
import hickle as hkl
from function import compute_train_ssim, saveimages, compute_mssim
from skimage.measure import compare_mse as mse
import os
from Model_ALL import build_gan,build_argcPredNet

# Data files
train_file = hkl.load(os.path.join('DATA_DIR', 'X_train.hkl')) # shape = (sequence_num,frames,img_rows, img_cols, channels)  The number of frames in each batch is 12
val_file = hkl.load(os.path.join('DATA_DIR', 'X_val.hkl'))     # shape = (one_sequence,frames,img_rows, img_cols, channels)


# Training parameters
img_rows = 96
img_cols = 96
channels = 1
input_img_num = 5
img_shape = (img_rows, img_cols, channels)
latent_dim = (6,) + (img_rows, img_cols, channels)
d_input_shape = (img_rows, img_cols, 2)
BATCH_SIZE = 10
onece_epoch = round(train_file.shape[0] / BATCH_SIZE + 0.5)
total_epoch = 100 * onece_epoch
n_critic = 5
nt = 6


# GAN-argcPredNet train
def gan_train(g, d, critic_model, generator_model, BATCH_SIZE, img_rows, img_cols, input_img_num):
    valid = -np.ones((BATCH_SIZE, 1))
    fake = np.ones((BATCH_SIZE, 1))
    dummy = np.zeros((BATCH_SIZE, 1))
    for epoch in range(0, total_epoch):
        for frame in range(7):
            for _ in range(n_critic):
                idx = np.random.randint(0, train_file.shape[0], BATCH_SIZE)
                imgs = train_file[idx]
                noise = imgs[:, frame: frame + input_img_num + 1]
                image_batch = imgs[:, frame + input_img_num]
                image_batch = np.concatenate([image_batch, image_batch], axis=3)
                d_loss = critic_model.train_on_batch([image_batch, noise], [valid, fake, dummy])

            g_loss = generator_model.train_on_batch(noise, valid)
        if frame % 10 == 0 and epoch % 5 == 0:
            compute_train_ssim(g, val_file, epoch, frame, d_loss[0], -g_loss, BATCH_SIZE, 7, img_rows, img_cols, COMPARE_DIR )
        if epoch % 5 == 0:
            g.save_weights(WEIGHTS_DIR + 'Generator'+ str(epoch)+ '.h5')
            d.save_weights(WEIGHTS_DIR + ' Discriminator'+ str(epoch)+ '.h5')


# argcPredNet train
def common_train(model, img_rows, img_cols, BATCH_SIZE):
    test_noise = np.concatenate([val_file[:, 0: 5], np.zeros((val_file.shape[0], 1, img_rows, img_cols, 1))], axis=1)
    image2 = val_file[0, 5, :, :, 0] * 255
    for epoch in range(total_epoch):
        test_gen_img = model.predict(test_noise, verbose=0)
        image1 = test_gen_img[0] * 255
        saveimages(image2, image1[5, :, :, 0],COMPARE_DIR)
        for i in range(7):
            input_img = np.concatenate([train_file[:,i:5+i], np.zeros((train_file.shape[0], 1, img_rows, img_cols, 1))], axis=1)
            loss = model.fit(input_img, train_file[:, i:6+i], batch_size=BATCH_SIZE, epochs=1, verbose=0)
        image1 = image1/255
        image2 = image2/255
        mse_s = mse(image2, image1[5, :, :, 0])
        ssim_s = compute_mssim(image2, image1[5, :, :, 0])
        print("[epoch %s][loss : %f] [MSE : %f] [SSIM : %f]" % (epoch, loss.history['loss'][0], mse_s, ssim_s))
        model.save_weights( WEIGHTS_DIR + str(epoch) + '.h5')

# GAN-argcPredNet train
d, g, critic_model, generator_model = build_gan(BATCH_SIZE, d_input_shape, latent_dim ,img_shape, img_rows, img_cols, input_img_num)
gan_train(g, d, critic_model, generator_model, BATCH_SIZE, img_rows, img_cols, input_img_num)

# argcPredNet train
model = build_argcPredNet(nt, img_rows, img_cols)
common_train(model, img_rows, img_cols, BATCH_SIZE)

