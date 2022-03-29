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
def crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    x,y,w,h = cv2.boundingRect(cnt)
    dst = img[y:y+h, x:x+w]
    return dst


@app.route("/predict" ,methods = ['GET' , 'POST'])
def segment():
    try:
        requestBody =request.data.decode('UTF-8')
        requestObject = json.loads(requestBody)
        base64Image = requestObject["base64"]
        base64_img_bytes = base64Image.encode('utf-8')
        with open('decoded_image.png', 'wb') as file_to_save:
            print(file_to_save)
            decoded_image_data = base64.decodebytes(base64_img_bytes) 
        image = Image.open(io.BytesIO(decoded_image_data))
        print(type(image))
        # image = Image.open(base64Image)
        cres = ColorProcess(image)
        x1 = cres[0]
        x2 = cres[1]
        x3 = cres[2]
        # imageio.imsave("withoutSymptom.jpg", x1)
        imageio.imsave("symptom.jpg", x2)
        # imageio.imsave("leaf.jpg", x3)
        print(type(x2))
        model = load_model('my_model5.h5')
        image = Image.fromarray(x2)
        print(type(image))
        image = image.resize(( 180,180))
        image = np.array(image)
        img_tensor = np.expand_dims(image, axis=0)
        prediction =  model.predict(img_tensor)
        

        maxClass = max(prediction[0])
        labelIndex =prediction[0].tolist().index(maxClass)
        print(prediction)

        # ['NitrogenHigh', 'NitrogenLow', 'PhosphorusHigh', 'PhosphorusLow', 'PotassiumHigh', 'PotassiumLow']  
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
    
        b64_string = base64.b64encode(x2)
        symptom = b64_string.decode('utf-8')

    
        b64_string = base64.b64encode(x3)
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


@app.route("/predictBlock" ,methods = ['GET' , 'POST'])
def block():
    try:
        requestBody =request.data.decode('UTF-8')
        requestObject = json.loads(requestBody)
        base64Image = requestObject["base64"]
        base64_img_bytes = base64Image.encode('utf-8')
        with open('decoded_image.png', 'wb') as file_to_save:
            print(file_to_save)
            decoded_image_data = base64.decodebytes(base64_img_bytes) 
        image = Image.open(io.BytesIO(decoded_image_data))

        cRes = ColorProcess(image)
        symptemImg = cRes[1]
        x1 = cRes[0]
        x2 = cRes[1]
        x3 = cRes[2]
        blockRes = devideToBlock(symptemImg)

        tip = blockRes[0]
        downO = blockRes[1]
        down1 = blockRes[2]
        down2 = blockRes[3]

        modelTip = load_model('tip_model.h5')
        model0 = load_model('0_model.h5')
        model1 = load_model('1_model.h5')
        model2 = load_model('2_model.h5')

     
        predictionTip =  modelTip.predict(getpredictImage(tip))
        prediction0 =  model0.predict(getpredictImage(downO))
        prediction1 =  model1.predict(getpredictImage(down1))
        prediction2 =  model2.predict(getpredictImage(down2))
        
        #['NitrogenHigh', 'NitrogenLow', 'PhosphorusHigh', 'PhosphorusLow', 'PotassiumHigh', 'PotassiumLow']
        print(predictionTip[0])
        print(prediction0[0])
        print(prediction1[0])
        print(prediction2[0])
        labels = [predictionTip[0].tolist().index(max(predictionTip[0])), prediction0[0].tolist().index(max(prediction0[0])), prediction1[0].tolist().index(max(prediction1[0])), prediction2[0].tolist().index(max(prediction2[0]))]
        labelIndex = most_frequent(labels)
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
    
        b64_string = base64.b64encode(x2)
        symptom = b64_string.decode('utf-8')

    
        b64_string = base64.b64encode(x3)
        leaf = b64_string.decode('utf-8')

        response = app.response_class(
        response='succeed',
        status=200,
        mimetype='application/json')
        response.headers.add('symptom',symptom)
        response.headers.add('leaf',leaf)
        response.headers.add('pred',labelIndex)
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

def devideToBlock(img):

    height = img.shape[0]
    width = img.shape[1]
    
    heightCutoff = height // 3
    tip = img[:heightCutoff, :]
    down = img[heightCutoff:,:]
    

    widthCutoff = width//3
    downO = down[:,:widthCutoff]
    downORest = down[:,widthCutoff:]
    down1 = downORest[:,:widthCutoff]
    down2 = downORest[:,widthCutoff:]
    return [tip,downO,down1,down2]

def getpredictImage(image):
    image = Image.fromarray(image)
    image = image.resize(( 180,180))
    image = np.array(image)
    img_tensor = np.expand_dims(image, axis=0)
    return img_tensor



def most_frequent(List):
    return max(set(List), key = List.count)


def ColorProcess(image):
    try :
        image = image.resize((180, 180))
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
        upper_mask = leaf_hsv[:,:,0] < 0.2 
        saturation_mask = leaf_hsv[:,:,1] > 0
        tipMask =upper_mask*lower_mask*saturation_mask

        lower_mask = leaf_hsv[:,:,0] > 0.2
        upper_mask = leaf_hsv[:,:,0] < 0.3 
        saturation_mask = leaf_hsv[:,:,1] > 0

        
        otherMask =upper_mask*lower_mask*saturation_mask
        mask2 = tipMask +otherMask  
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

        return [x1 , x2 , x3]

    except Exception as e:
        return 0
