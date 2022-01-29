from flask import Flask
from flask import request
import imageio
from skimage.color import rgb2hsv
import numpy as np
import cv2 
app = Flask(__name__)


@app.route("/")
def hello():
    im1 = imageio.imread("C:/newfolder/image.jpg")
    bags_hsv = rgb2hsv(im1)
   

    lower_mask = bags_hsv[:,:,0] > 0.2 #refer to hue channel (in the colorbar)
    upper_mask = bags_hsv[:,:,0] < 0.5 #refer to transparency channel (in the colorbar)
    saturation_mask = bags_hsv[:,:,1] > 0.4 
 
    mask1 = upper_mask*lower_mask*saturation_mask
    red = im1[:,:,0]*mask1
    green = im1[:,:,1]*mask1
    blue = im1[:,:,2]*mask1
    bags_masked = np.dstack((red,green,blue))
    x1=cv2.fastNlMeansDenoisingColored(bags_masked,None,10,10,7,21)


    lower_mask = bags_hsv[:,:,0] > 0.0 #refer to hue channel (in the colorbar)
    upper_mask = bags_hsv[:,:,0] < 0.18 #refer to transparency channel (in the colorbar)
    saturation_mask = bags_hsv[:,:,1] > 0
    mask2 =upper_mask*lower_mask*saturation_mask
    red = im1[:,:,0]*mask2
    green = im1[:,:,1]*mask2
    blue = im1[:,:,2]*mask2
    bags_masked = np.dstack((red,green,blue))
    x2=cv2.fastNlMeansDenoisingColored(bags_masked,None,10,10,7,21)


    mask3 = mask1 + mask2
    red = im1[:,:,0]*mask3
    green = im1[:,:,1]*mask3
    blue = im1[:,:,2]*mask3
    bags_masked = np.dstack((red,green,blue))
    x3=cv2.fastNlMeansDenoisingColored(bags_masked,None,10,10,7,21)
    imageio.imsave("C:/newfolder/image1.jpg", x1)
    imageio.imsave("C:/newfolder/image2.jpg", x2)
    imageio.imsave("C:/newfolder/image3.jpg", x3)

    #imshow(x)
    return "Hello, Farmers!"
    
    #bags_masked = np.dstack((red,green,blue))
    #imshow(bags_masked)



#refer to hue channel (in the colorbar)
    # lower_mask = bags_hsv[:,:,0] > -0.1 #refer to hue channel (in the colorbar)
    # upper_mask = bags_hsv[:,:,0] < 0.2 #refer to transparency channel (in the colorbar)
    # saturation_mask = bags_hsv[:,:,1] > 0
    # mask1 =upper_mask*lower_mask*saturation_mask
    # mask2 = mask1+ mask
    # red = im1[:,:,0]*mask2
    # green = im1[:,:,1]*mask2
    # blue = im1[:,:,2]*mask2
    # asd =mask+mask2
    # bags_masked = np.dstack((red,green,blue))
    # x=cv2.fastNlMeansDenoisingColored(bags_masked,None,10,10,7,21)



@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    
    if request.method == 'POST':
        f = request.files['the_file']
        f.save('as.jpg')
        return "work"
    else:
        return "Uploads not work"

app.run(host='127.0.0.1', port=8050)


