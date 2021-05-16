# Description
This paper introduces a Generative Adversarial Networks, GAN-argcPredNet, which can be used for radar-based precipitation nowcasting.
This network is trained to be a prediction model that extrapolates the next 7 frames from the first 5 frames.<br>
The source code for the GAN-argcPredNet model written using the Keras functional API is in the model.py file.
# GAN-argcPredNet
This model references the depth coding structure of the prednetmodel proposed by bill-lotter, and is based on the rgcLSTM design idea of NellyElsayed.
# Radar data
The experimental data is the radar puzzle data of Shenzhen area provided by Guangdong Meteorological Bureau. It does not support the open sharing.
# Train
Train.py is our training file. This step will train the network into a model that is extrapolated from 5 frames to 7 frames. This includes the training of argcPrdNet and GAN-argcPredNet two networks.
## GAN-argcPredNet
    g.save_weights(WEIGHTS_DIR+ 'Generator’+ str(epoch)+’.h5')  d.save_weights(WEIGHTS_DIR+ ' Discriminator’+ str(epoch)+’.h5') <br>
save the weight files of the generator and the discriminator respectively. During the training process, the compute_train_ssim function is responsible for generating and saving the comparison image of the training process, and displaying the training loss value.Save the weight files of the generator and the discriminator respectively. During the training process, the compute_train_ssim function is responsible for generating and saving the comparison image of the training process, and displaying the training loss value.
## argcPrdNet
    model.save_weights (WEIGHTS_DIR + str(epoch) + '.h5') <br>
Save the weight file generated during the training process.
# Prediction
   The prediction code is stored in the predict.py file, 
    X_test = hkl.load(TEST_DIR)/255 loads the test set file, 
    model.load_weights(WEIGHTS_DIR) loads the trained weight file, 
Then through GanpredictMulImage(), test_predictImag() functions respectively generate prediction data, and finally save the prediction data by hkl.dump(preimage, PREDICT_DIR).
# NOTE
You can cite the GAN-argcPredNet model repository as follows:<br>
Url:https://github.com/LukaDoncic0/GAN-argcPredNET/edit/main/README.md<br>
BibTeX:<br>
@misc{ Liu Yan2021GAN-argcPredNet,<br>
  author = {Liu Yan},<br>
  title = { A Generative Adversarial Model for Radar Echo Extrapolation Based on Convolutional Recurrent Units<br>
},<br>
  year = {2021},<br>
  publisher = {GitHub},<br>
  journal = {GitHub repository},<br>
  howpublished = {\url{ }}<br>
}<br>
