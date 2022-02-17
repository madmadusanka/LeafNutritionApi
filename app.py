import base64
from io import BytesIO
from typing import NamedTuple
from flask import Flask, render_template, request, url_for, jsonify
import imageio
from skimage.color import rgb2hsv
import numpy as np
import cv2 
from keras.models import load_model
from PIL import Image
import json
app = Flask(__name__)


class Request():

    def __init__(self, base64):

        self.base64 = base64

@app.route("/predict" ,methods = ['GET' , 'POST'])
def hello():
    try:
        print("request.get_json()")

        base64 = request.headers.get('base64')
        # img = Image.open(BytesIO(base64.decodebytes(base64)))
        print(base64)

        base64_img_bytes = base64.encode('utf-8')
        with open('decoded_image.png', 'wb') as file_to_save:
            decoded_image_data = base64.decodebytes(base64_img_bytes)
            file_to_save.write(decoded_image_data)
   
        # with open("imageToSave.png", "wb") as fh:
        #     fh.write(base64.b64decode(text))

        im1 = imageio.imread("C:/newfolder/image.jpg")
        leaf_hsv = rgb2hsv(im1)
    

        lower_mask = leaf_hsv[:,:,0] > 0.2 #refer to hue channel (in the colorbar)
        upper_mask = leaf_hsv[:,:,0] < 0.5 #refer to transparency channel (in the colorbar)
        saturation_mask = leaf_hsv[:,:,1] > 0.4 
    
        mask1 = upper_mask*lower_mask*saturation_mask
        red = im1[:,:,0]*mask1
        green = im1[:,:,1]*mask1
        blue = im1[:,:,2]*mask1
        leaf_masked = np.dstack((red,green,blue))
        x1=cv2.fastNlMeansDenoisingColored(leaf_masked,None,10,10,7,21)


        lower_mask = leaf_hsv[:,:,0] > 0.0 #refer to hue channel (in the colorbar)
        upper_mask = leaf_hsv[:,:,0] < 0.18 #refer to transparency channel (in the colorbar)
        saturation_mask = leaf_hsv[:,:,1] > 0
        mask2 =upper_mask*lower_mask*saturation_mask
        red = im1[:,:,0]*mask2
        green = im1[:,:,1]*mask2
        blue = im1[:,:,2]*mask2
        leaf_masked = np.dstack((red,green,blue))
        x2=cv2.fastNlMeansDenoisingColored(leaf_masked,None,10,10,7,21)


        mask3 = mask1 + mask2
        red = im1[:,:,0]*mask3
        green = im1[:,:,1]*mask3
        blue = im1[:,:,2]*mask3
        leaf_masked = np.dstack((red,green,blue))
        x3=cv2.fastNlMeansDenoisingColored(leaf_masked,None,10,10,7,21)
        imageio.imsave("C:/newfolder/image1.jpg", x1)
        imageio.imsave("C:/newfolder/image2.jpg", x2)
        imageio.imsave("C:/newfolder/image3.jpg", x3)




















        model = load_model('E:/User Document/my_model.h5')
        image = Image.open('E:/User Document/processed/PotassiumHigh/PotassiumHigh44.jpg')
        image = image.resize((180, 180))
        print("this")
        print(image)
        image = np.array(image)
        img_tensor = np.expand_dims(image, axis=0) 
        print(model.predict(img_tensor))
    except Exception as e:
        print(e)
        return e
    return 'fail'
    
  



@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    
    if request.method == 'POST':
        f = request.files['the_file']
        f.save('as.jpg')
        return "work"
    else:
        return "Uploads not work"

app.run(host='127.0.0.1', port=8050)


