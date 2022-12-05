import socket
import cv2
import time
import numpy as np
from AlphaBot import AlphaBot
from picamera import PiCamera
from bottle import Bottle, get, run, request

app = Bottle()

def setup_app(app):
    app.bot = AlphaBot()
    app.camera = PiCamera()
    app.camera.resolution = (320,240)
    app.camera.framerate = 24
    app.camera.start_preview()
    time.sleep(2)
        
setup_app(app)

@app.get("/camera/get")
def get_image():
    image = np.empty((240 * 320 * 3,), dtype=np.uint8)
    app.camera.capture(image, 'bgr', use_video_port=True)
    image = image.reshape((240, 320, 3))
    return cv2.imencode('.jpg', image)[1].tobytes()
    
@app.get('/robot/set/velocity')
def set_velocity():
    vel = request.query.value.split(',')
    app.bot.setMotor(float(vel[0]), float(vel[1]))
    if request.query.time != '':
        time_run = float(request.query.time)
        time.sleep(time_run)
        app.bot.stop()
    

s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.connect(('8.8.8.8',80))
localhost=s.getsockname()[0]
run(app, host=localhost, port=8000)