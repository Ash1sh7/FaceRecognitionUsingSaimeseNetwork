import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import siamese_network as SN

from PIL import Image

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from time import gmtime, strftime
import helper

PROJ_PATH = os.getcwd()
FACE_DTA_PATH = PROJ_PATH + "\\dataset\\faces"
PATH_LIST = os.listdir(FACE_DTA_PATH)

MODEL_PATH = PROJ_PATH +  "\\model\\siamese-face-model.h5"
#MODEL_NAME = 'siamese-face-model.h5'

CAT_LIST = ["Harrison Ford",
           "Ed Westwick",
           "Chris Hemsworth",
           "Nicole Kidman",
           "Shah Rukh Khan",
           "Roger Moore",
           "Jason Statham",
           "Megan Fox",
           "Marisa Tomei",
           "Nicolas Cage",
           "Britney Spears",
           "Reese Witherspoon"]

def load_my_model():
    #model_path = os.path.join(MODEL_PATH, MODEL_NAME)
    model_path = MODEL_PATH
    global model
    model = load_model(model_path,
                       custom_objects={'contrastive_loss': SN.contrastive_loss})
    print(' * My model loaded.....')

app = Flask(__name__)

UPLOAD_FOLDER                       =   './upload_images/'
if not os.path.exists(UPLOAD_FOLDER):
  os.mkdir(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS                  =   set(['png', 'bmp', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER']         =   UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH']    =   1 * 600 * 600

@app.route('/')
def upload_file():
   return render_template('upload.html')

def make_prediction(file_path):
	print(' * Starting prediction.....')
	ref_image = helper.get_image_from_filename(file_path)

	# find the category
	results = []
	cat = 0
	for cur_path in PATH_LIST:
		filelist = os.listdir(os.path.join(FACE_DTA_PATH, cur_path))
		idx = np.random.randint(len(filelist))
		cur_image = helper.get_image(FACE_DTA_PATH, cat, idx)
		dist = model.predict([ref_image, cur_image])[0][0]
		results.append(dist)
		cat += 1

	idx = np.argmin(results)

	return CAT_LIST[idx]

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
   if request.method == 'POST':
       f = request.files['file']
       filename_saved = secure_filename(f.filename)
       
       # if no image file selected, stay upload.html page
       if len(filename_saved) == 0:
           return render_template('upload.html')

       # extract file extension
       file_ext = filename_saved.split('.')[-1]

       filename_saved = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
       filename_saved = "%s.%s" % (filename_saved, file_ext)
       filename_saved = os.path.join(UPLOAD_FOLDER, filename_saved)

       if os.path.isfile(filename_saved):
           os.remove(filename_saved)

       f.save(filename_saved)

       # extract faces
       print("....... Now extracting a face .......")
       pixels = helper.extract_face(filename_saved, required_size=(160, 160))
       img = Image.fromarray(pixels, mode='RGB')
       tmp_filename = os.path.join(UPLOAD_FOLDER, 'tmp_rgb.jpg')
       img.save(tmp_filename)

       # convert to grayscale
       print("....... Now converting the image .......")
       img = cv2.imread(tmp_filename, cv2.IMREAD_GRAYSCALE)
       target_path = os.path.join(UPLOAD_FOLDER, 'tmp_gray.jpg')
       cv2.imwrite(target_path, img)

       # predict
       print("....... Now predicting the face .......")
       ret_val = make_prediction(target_path)
       ret_val = "Predicted: %s" % ret_val
       return render_template('disp_result.html',
                              retval=ret_val,
                              origianl_image=filename_saved,
                              extracted_image=tmp_filename)

if __name__ == '__main__':
    load_my_model()
    app.run(debug=True, threaded=False)