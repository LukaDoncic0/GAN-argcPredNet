import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
import numpy as np
from keras.layers.merge import _Merge
from keras import backend as K



def saveimages(image1, image2, save_name):
    plt.subplot(121)
    plt.imshow(image1, interpolation='none')
    plt.subplot(122)
    plt.imshow(image2, interpolation='none')
    plt.savefig(save_name)

def simple_slice(x, index):
    return x[:, index, :, :]

def subimg(x, index1, index2):
    return x[:, index1: index2, :, :]

def compute_mssim(im_list1, im_list2):
    ssim_s = 0.
    for i in range(im_list1.shape[0]):
        ssim_s += ssim(im_list1[i], im_list2[i])
    return ssim_s / im_list1.shape[0]

def predictImage(image, FRAME, model, batch_size, img_rows, img_cols):
    preimage = image[:, 0:5, :, :, 0]
    nosie = np.zeros((image.shape[0], 1, img_rows, img_cols, 1))
    for i in range(FRAME-1):
        input_img = np.concatenate([np.reshape(preimage[:, i: i + 5], newshape=(image.shape[0], 5, img_rows, img_cols, 1)), nosie],axis=1)
        result = model.predict(input_img, batch_size)
        preimage = np.concatenate((preimage, np.reshape(result[:, :, :, 1], newshape=(image.shape[0], 1, img_rows, img_cols))),axis=1)
    return preimage * 255.

def compute_score_train(data, model, batch_size, FRAME, img_rows, img_cols):
    preimage = predictImage(data, FRAME, model, batch_size, img_rows, img_cols)/255
    mse_s = mse(data[:, 5:, :, :, 0], preimage[:, 5:])
    ssim_s = compute_mssim(data[:, 5:, :, :, 0], preimage[:, 5:])
    return mse_s, ssim_s, preimage[0, 5]


def compute_train_ssim(model, val_imgs, epoch, i, d_loss, g_loss, batch_size, FRAME, img_rows, img_cols, save_dir):
    mse_s, ssim_s, gen_img = compute_score_train(val_imgs, model, batch_size, FRAME, img_rows, img_cols)
    print("[epoch %d][frame : %d][MSE : %f] [SSIM : %f] [D loss: %f] [G loss: %f]" % (epoch, i, mse_s, ssim_s,
                                                                                      d_loss, g_loss))
    image2 = val_imgs[0, 5, :, :, 0] * 255
    saveimages(image2, gen_img,save_dir + str(epoch) + "_" + str(i) + ".png")
    return ssim_s, mse_s


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((10, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def extrap_loss(y_true, y_hat):
    y_true = y_true[:, 1:]
    y_hat = y_hat[:, 1:]
    return 0.5 * K.mean(K.abs(y_true - y_hat), axis=-1)


def test_predictImage(test_img, model, BATCH_SIZE, FRAME, img_rows, img_cols):
    preimage = test_img[:, 0:5]
    for i in range(FRAME):
        input_img = np.concatenate([preimage[:, 0+i:5+i], np.zeros((input_img.shape[0], 1, img_rows, img_cols, 1))], axis=1)
        pre_img = model.predict(input_img, BATCH_SIZE)
        preimage = np.concatenate([preimage, np.reshape(pre_img[:, 5, :, :, 0], newshape=(input_img.shape[0], 1, img_rows, img_cols))], axis=1)
    return preimage * 255.








