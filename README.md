# Description
This paper introduces a Generative Adversarial Networks, GAN-argcPredNet v1.0, which can be used for radar-based precipitation nowcasting.<br><br>
The  generator is stored in the `argc_PredNet.py file`, and the completion process of building GAN-argcPredNet is in the `GAN-argcPredNet_Train.py` file.<br><br>
This network is trained to be a prediction model that extrapolates the next 7 frames from the first 5 frames.<br><br>


# GAN-argcPredNet v1.0
This model references the depth coding structure of the prednetmodel proposed by [bill-lotter](https://github.com/coxlab/prednet), and is based on the rgcLSTM design idea of [NellyElsayed](https://github.com/NellyElsayed/rgcLSTM).We modified the recurrent unit in the network and used it as a generator.In addition, a two-channel input network with four layers of convolution is the discriminator of GAN-argcPredNet.<br><br>
![image](https://github.com/LukaDoncic0/GAN-argcPredNet/blob/main/png/model.tif)

# Radar data
The experimental data is the radar mosaic of Shenzhen area provided by Guangdong Meteorological Bureau. It does not support the open sharing.For data access, please contact Kun Zheng (ZhengK@cug.edu.cn) and Yan Liu (liuyan_@cug.edu.cn).
# Train
The files of the training model are stored in the `GAN-argcPredNet_Train.py` and `argcPredNet_Train.py` files. The advantages of GAN-argcPredNet can be seen through the training experiments on these two models.
## GAN-argcPredNet
Save the weight files of the generator and the discriminator respectively:<br>

    self.generator.save_weights(weights_dir+'gen_weight'+epoch + '.h5')
    
    self.critic.save_weights(weights_dir+'dis_weight' + epoch + '.h5')
During the training process, the `compute_train_ssim()` function is responsible for generating and saving the comparison image of the training process, and displaying the training loss value.
## argcPrdNet
The `Saveimages()` function will save the contrast images that are constantly changing during the training process.
Save the weight file generated during the training process:

    model.save_weights (WEIGHTS_DIR + str(epoch) + '.h5') 
# Prediction
The prediction code is stored in the `predict.py` file.  `X_test = hkl.load(TEST_DIR)` loads the test set file, `model.load_weights(WEIGHTS_DIR)` loads the trained weight file.
Then through `test_predictImag() `functions respectively generate prediction data.Finally save the prediction data by: 

    hkl.dump(Preimage, PREDICT_DIR).
# NOTE
You can cite the GAN-argcPredNet model repository as follows:<br>
https://github.com/LukaDoncic0/GAN-argcPredNet<br>
# Reference
@article{elsayed2018reduced,<br><br>
title={Reduced-Gate Convolutional LSTM Using Predictive Coding for Spatiotemporal Prediction},<br><br>
author={Elsayed, Nelly and Maida, Anthony S and Bayoumi, Magdy},<br><br>
journal={arXiv preprint arXiv:1810.07251},<br><br>
year={2018}<br><br>
}<br><br>
AND<br><br>
@article{lotter2016deep,<br><br>
title={Deep predictive coding networks for video prediction and unsupervised learning<br><br>
author={Lotter, William and Kreiman, Gabriel and Cox, David},<br><br>
journal={arXiv preprint arXiv:1605.08104},<br><br>
year={2016}<br><br>
}<br><br>
