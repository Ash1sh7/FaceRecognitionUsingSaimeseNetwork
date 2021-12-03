import os
import numpy as np
import tensorflow as tf
import datetime
import siamese_network as SN
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#print(np.__version__)

proj_path = os.getcwd()
DATA_PATH = proj_path + '\\dataset\\numpy'
X_data_name = 'X_data.npy'
Y_data_name = 'Y_data.npy'

CHKPT_PATH = proj_path + '\\checkpoint'
CHK_PT_NAME = 'chkpt-model.ckpt'

MODEL_PATH = proj_path + '\\model'
MODEL_FILENAME = 'siamese-face-model.h5'

TEST_SIZE = .20
NUM_EPOCHS = 50
BATCH_SIZE = 16
PATIENCE = 10

X = np.load(os.path.join(DATA_PATH, X_data_name))
Y = np.load(os.path.join(DATA_PATH, Y_data_name))

print(X.shape)
print(Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE)

input_dim = x_train.shape[2:]
print(len(input_dim))

img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)

base_network = SN.build_base_network(input_dim)
base_network.summary()

feat_vecs_a = base_network(img_a)
feat_vecs_b = base_network(img_b)

print(feat_vecs_a)
print(feat_vecs_b)

#distance = Lambda(SN.euclidean_distance, output_shape=SN.eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])

#distance = SN.euclidean_distance([feat_vecs_a, feat_vecs_b])

earlystopper = EarlyStopping(patience=PATIENCE, verbose=1)

if not os.path.exists(CHKPT_PATH):
  os.mkdir(CHKPT_PATH)
  
checkpt_path = os.path.join(CHKPT_PATH, CHK_PT_NAME)
print('Check Point File Path: {}'.format(checkpt_path))

checkpointer = ModelCheckpoint(checkpt_path, verbose=1, save_best_only=True)

#rms = RMSprop()
model = Model(inputs=[img_a, img_b], outputs=SN.euclidean_distance([feat_vecs_a, feat_vecs_b]))
model.summary()

model.compile(loss=SN.contrastive_loss, optimizer='rmsprop', metrics=['accuracy'])

log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

img_1 = x_train[:, 0]
img_2 = x_train[:, 1] 

# elapsed time checking
import time
start_time = time.time()

hist = model.fit([img_1, img_2],
                 y_train,
                 validation_split=TEST_SIZE,
                 batch_size=BATCH_SIZE,
                 verbose=2,
                 epochs=NUM_EPOCHS,
                 callbacks=[earlystopper, checkpointer, tensorboard_callback])

print('\n\n Elapsed Time for Training: {} seconds ---'.format(time.time() - start_time))

if not os.path.exists(MODEL_PATH):
  os.mkdir(MODEL_PATH)

model.save(os.path.join(MODEL_PATH, MODEL_FILENAME))

for key in ['loss', 'val_loss']:
  plt.plot(hist.history[key], label=key)

plt.legend()
plt.show()

pred = model.predict([x_test[:, 0], x_test[:, 1]])

print("Accuracy: ", SN.compute_accuracy(pred, y_test))