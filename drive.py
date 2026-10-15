import socketio
import base64
import matplotlib.image as mpimg
from flask import Flask
from io import BytesIO
import numpy as np
import keras
import cv2
import eventlet.wsgi

sio = socketio.Server() # create server
app = Flask(__name__) # create app that server attaches to
model = keras.models.load_model('model.keras')

max_speed = 20

# redefine preprocess function
def preprocess(image):
    # mpimg.imread() returns list of shape (height, width, size) in this case (160, 320, 3) because 160 rows (height), 320 columns (width), and 3 color chanels in each column in the row [R, G, B]
    image = image[60:135, :, :] # cut off top and a little bottom
    # kernal = grid of pixels GuassianBlur() uses to avg pixel values, i.e. bigger kernal = more blurry smaller = less
    image = cv2.GaussianBlur(image, (3,3), 0) # (image, blur kernal, blur strength (0=dynamic based on kernal))
    image = cv2.resize(image, (200, 66)) # resize(image, (width, height)) width is 320 and height is ~70 after cropping but was 160 originally
    image = image / 255 # every pixel value in array between 0 and 1. 255 since they are 8 bit

    return image

@sio.on('telemetry') # runs every time a message named 'telemetry' comes on
def telemetry(sid, data):
    throttle = 1.0 - float(data['speed']) / max_speed
    image = mpimg.imread(
        # data['image'] returns text which needs to be converted to bytes with
        # b64decode, but in order for mpimg.imread() to turn that into a pixel
        # array, it needs a file to perform .imread() so BytesIO makes the bytes
        # act like a local file that can have read() and things be done to them
        BytesIO(base64.b64decode(data['image'])),
        # training images are jpg and jpg/png are decoded into pixel values
        # differently, so need to specifiy jpg since imread() defaults to png
        format='jpg'
        )
    image = preprocess(image)
    # make shape (1, 66, 200, 3) instead of (66, 200, 3) because keras model.predict()
    # expects the batch size to be the first number in the shape so since we are taking
    # in one image at a time we just wrap it in an array like that with batch
    # size 1. without this line it would treat it as 66 batches each with shape
    # (200, 3) instead of 1 batch of size (66, 200, 3) which is what we want
    image = np.array([image]) # ^

    steering_angle = float(model.predict(image, verbose=1))

    send_control(steering_angle, throttle) # send control results to udacity

def send_control(steering_angle, throttle):
    sio.emit('steer', # emit message with name steer ( udacity is listening for message with that exact name )
             data={ # data packet. udacity receives this dictionary and applies the values passed to the car at that moment
                    'steering_angle': str(steering_angle),
                    'throttle': str(throttle)
                }
            )

if __name__ == '__main__':
    # wrap socket io server and Flask app into one app
    app = socketio.WSGIApp(sio, app) # WSGI = Web Server Gateway Interface. Standard way python web servers run apps
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)