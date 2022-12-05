# access each wheel and the camera onboard of Alphabot

import numpy as np
import requests
import cv2 
# mtx = np.array([[1.06374294e+3, 0, 3.52330611e+02],[0,1.06561853e+03,2.57858244e+02],[0,0,1]])
# dist = np.array([-4.24422461e-01,-5.78906352e-01,-9.31874026e-04,4.24078376e-05,5.26953417e+00])

class Alphabot:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.wheel_vel = [0, 0]

    ##########################################
    # Change the robot velocity here
    # tick = forward speed
    # turning_tick = turning speed
    ########################################## 
    def set_velocity(self, command, tick=10, turning_tick=5, time=None): 
        l_vel = command[0]*tick - command[1]*turning_tick
        r_vel = command[0]*tick + command[1]*turning_tick
        self.wheel_vel = [l_vel, r_vel]
        # print(self.wheel_vel)
        if time == None:
            requests.get(
                f"http://{self.ip}:{self.port}/robot/set/velocity?value="+str(l_vel)+","+str(r_vel))
        elif time == 0:
            pass
        else:
            assert (time > 0), "Time must be positive."
            assert (time < 30), "Time must be less than network timeout (20s)."
            requests.get(
                "http://"+self.ip+":"+str(self.port)+"/robot/set/velocity?value="+str(l_vel)+","+str(r_vel)
                            +"&time="+str(time))
        return l_vel, r_vel
        
    def get_image(self):
        try:
            r = requests.get(f"http://{self.ip}:{self.port}/camera/get")
            img = cv2.imdecode(np.frombuffer(r.content,np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # h,  w = img.shape[:2]
            # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            # # undistort
            # img = cv2.undistort(img, mtx, dist, None, newcameramtx)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("Image retrieval timed out.")
            img = np.zeros((240,320,3), dtype=np.uint8)
        return img
