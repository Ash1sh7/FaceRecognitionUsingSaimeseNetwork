import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import siamese_network as SN

PROJ_PATH = os.getcwd()
MODEL_PATH = PROJ_PATH +  "\\model"
MODEL_NAME = "siamese-face-model.h5"            #Enter .h5 file name
TFLITE_FILE_NAME = 'siamese-face-model.tflite'  #Enter .tflite file name you wish to save as

## load the model
model = load_model(os.path.join(MODEL_PATH, MODEL_NAME), 
                   custom_objects={'contrastive_loss': SN.contrastive_loss})

tflite_file  = os.path.join(MODEL_PATH, TFLITE_FILE_NAME)
print("Saving tflite_file as" + tflite_file)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

#now save the tflite model to the file
#tflite_model.save(tflite_file)   #Note this does not seem to work although in google documentation
open(tflite_file, "wb").write(tflite_model)