import hickle as hkl
from Model_ALL import build_argcPredNet
from function import test_predictImage
import os

# Test parameters
img_rows = 96
img_cols = 96
channels = 1
input_img_num = 5
FRAME = 7
img_shape = (img_rows, img_cols, channels)
latent_dim = (6,) + (img_rows, img_cols, channels)
BATCH_SIZE = 10
nt = 6

#Load test data
X_test = hkl.load(os.path.join('DATA_DIR', 'X_train.hkl'))/255

# Predict
model = build_argcPredNet(nt, img_rows, img_cols)
model.load_weights( WEIGHTS_DIR )
Preimage = test_predictImage(X_test, model, BATCH_SIZE, FRAME, img_rows, img_cols)
hkl.dump(Preimage, PREDICT_DIR)
print("End！")




