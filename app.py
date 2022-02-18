import base64
from io import BytesIO
import io
from typing import NamedTuple
from urllib import response
from flask import Flask, render_template, request, url_for, jsonify
import imageio
from matplotlib.pyplot import clf
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
  
def toBase64(binary_file):
    binary_file_data = binary_file
    base64_encoded_data = base64.b64encode(binary_file_data)
    base64_message = base64_encoded_data
    return base64_message

@app.route("/predict" ,methods = ['GET' , 'POST'])
def hello():
    try:
        print("request.get_json()")
        base64Image = request.headers.get('base64') 
        base64_img_bytes = base64Image.encode('utf-8')
        with open('decoded_image.png', 'wb') as file_to_save:
            print(file_to_save)
            decoded_image_data = base64.decodebytes(base64_img_bytes)
            file_to_save.write(decoded_image_data)  
        image = Image.open(io.BytesIO(decoded_image_data))
        image_np = np.array(image)
    
        im1 = image_np
        leaf_hsv = rgb2hsv(im1)
  
        lower_mask = leaf_hsv[:,:,0] > 0.2 
        upper_mask = leaf_hsv[:,:,0] < 0.5 
        saturation_mask = leaf_hsv[:,:,1] > 0.4 
    
        mask1 = upper_mask*lower_mask*saturation_mask
        red = im1[:,:,0]*mask1
        green = im1[:,:,1]*mask1
        blue = im1[:,:,2]*mask1
        leaf_masked = np.dstack((red,green,blue))
        x1=cv2.fastNlMeansDenoisingColored(leaf_masked,None,10,10,7,21)
        


        lower_mask = leaf_hsv[:,:,0] > 0.0 
        upper_mask = leaf_hsv[:,:,0] < 0.18 
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
        
        black_pixels = np.where(
            (x1[:, :, 0] == 0) & 
            (x1[:, :, 1] == 0) & 
            (x1[:, :, 2] == 0)
            )
        x1[black_pixels] = [255, 255, 255]
        
        black_pixels = np.where(
            (x2[:, :, 0] == 0) & 
            (x2[:, :, 1] == 0) & 
            (x2[:, :, 2] == 0)
            )
        x2[black_pixels] = [255, 255, 255]
        
        black_pixels = np.where(
            (x3[:, :, 0] == 0) & 
            (x3[:, :, 2] == 0)
            )
        x3[black_pixels] = [255, 255, 255]
        
        imageio.imsave("withoutSymptom.jpg", x1)
        imageio.imsave("symptom.jpg", x2)
        imageio.imsave("leaf.jpg", x3)

        model = load_model('E:/User Document/my_model.h5')
        image = Image.open('symptom.jpg')
        image = image.resize((180, 180))
        image = np.array(image)
        img_tensor = np.expand_dims(image, axis=0)
        prediction =  model.predict(img_tensor)
        maxClass = max(prediction[0])
        labelIndex =prediction[0].tolist().index(maxClass)

        # ['NitrogenHigh', 'NitrogenLow', 'PhosphorusHigh', 'PhosphorusLow', 'PotassiumHigh', 'PotassiumLow']  


        with open("symptom.jpg", "rb") as img_file:
            b64_string = base64.b64encode(img_file.read())
            symptom = b64_string.decode('utf-8')

        with open("leaf.jpg", "rb") as img_file:
            b64_string = base64.b64encode(img_file.read())
            leaf = b64_string.decode('utf-8')

        response = app.response_class(
        response='succeed',
        status=200,
        mimetype='application/json')
        response.headers.add('symptom',symptom)
        response.headers.add('leaf',leaf)
        response.headers.add('pred',labelIndex)
        print(labelIndex)
        return response
    except Exception as e:
        print(e)
        return "failed"
    return 'succeed'
    



@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    
    if request.method == 'POST':
        f = request.files['the_file']
        f.save('as.jpg')
        return "work"
    else:
        return "Uploads not work"

app.run(host='127.0.0.1', port=8050)


