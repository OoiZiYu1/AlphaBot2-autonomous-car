# teleoperate the robot and perform SLAM
# will be extended in following milestones for system integration

# basic python packages
from cmath import atan
import numpy as np
import cv2 
import os, sys
import time

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import Alphabot # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector

#Import Task3 Items
from pathlib import Path
import json

# import cv2
import math
from machinevisiontoolbox import Image
import statistics

import matplotlib.pyplot as plt
import PIL

class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = Alphabot(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.06) # size of the ARUCO markers ORIGINAL 0.06 #Use 0.054 For Home Testing

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 600
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        if args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.ckpt, use_gpu=False)
            self.network_vis = np.ones((240, 320,3))* 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')
        self.mtx = np.array([[1.07922768e+03,0.00000000e+00, 3.45066466e+02], [0.00000000e+00,1.07537074e+03,2.57634560e+02], [0,0,1]])
        self.dist = np.array([-4.12927114e-01,2.88401538e-01,3.56945894e-04,2.68483068e-03,5.40200655e-01])
        ### M4 Task1 ###
        self.order =0
        self.Complete_t_d =True
        self.lv, self.rv = [0,0]
        self.drive_time =[0]
        self.turn_time =[0]
        self.turntime_left_around =None 
        self.turntime_left = None
        self.drivetime_left = None
        self.waypoint = [0,0]
        self.nextwaypoint_bool = True
        self.not_scanning_around = True
        self.DrivingCommand = False
        self.TurningCommand = False

                ### Task2 ###
        self.fruits_true_pos = []
        self.aruco_true_pos=[]
        self.start = False
        self.around_order =0
        self.interlock_turn_around = True
        self.points =[]
        self.TruePositions = []
        
        ### M4 Task3 ###
        self.fruits_list = []
        self.search_list = []
        self.numFruitList = []
        self.FruitsToNum_dict = {"redapple": 1, "greenapple": 2, "orange": 3, "mango": 4, "capsicum": 5}
        self.NumToFruits_dict = {v: k for k, v in self.FruitsToNum_dict.items()}
        self.camera_matrix = []
        self.RegenComplete = False
        self.RegenerateFlag = False

        self.RedapplePics = []
        self.GreenapplePics = []
        self.OrangePics = []
        self.MangoPics = []
        self.CapsicumPics = []

        self.RedappleFound = False
        self.GreenappleFound = False
        self.OrangeFound = False
        self.MangoFound = False
        self.CapsicumFound = False
        self.aligned = False

        self.RedapplePose = []
        self.GreenapplePose = []
        self.OrangePose = []
        self.MangoPose = []
        self.CapsicumPose = []

        self.fruits_true_pos = np.zeros([1,2])
        self.base_dir = Path('./')
        self.target_est ={}


        
    def Stationarycontrol(self):       
        if args.play_data:
            self.lv, self.rv = self.pibot.set_velocity()            
        else:
            self.lv, self.rv = self.pibot.set_velocity(
                self.command['motion'])            

        if not self.data is None:
            self.data.write_keyboard(self.lv, self.rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(self.lv, self.rv, dt)
        self.control_clock = time.time()
        return drive_meas

    def control(self, waypoint, robot_pose):       
        fileS = "calibration/param/scale.txt"
        scale = np.loadtxt(fileS, delimiter=',') #Distance ticks/m
        fileB = "calibration/param/baseline.txt"
        baseline = np.loadtxt(fileB, delimiter=',') #angle ticks/rad

        wheel_lv = 25 # tick to move the robot
        wheel_lr = 30
        
        
        
        if args.play_data:
            self.lv, self.rv = self.pibot.set_velocity()            
        else:
            x_goal, y_goal = waypoint
            x, y, __ = [robot_pose[0],robot_pose[1],robot_pose[2]]
            print('robot pose')
            print([robot_pose])

            if robot_pose[2]>=(np.pi*2):
                theta = robot_pose[2]%(np.pi*2)
                if theta <= (np.pi*2) and theta > (np.pi):
                    theta = theta -np.pi*2
            elif robot_pose[2]<(-np.pi*2):
                theta = -(np.pi*2 - robot_pose[2]%(np.pi*2))+(np.pi*2)
                if theta <= (np.pi*2) and theta > (np.pi):
                    theta = theta-np.pi*2
            elif robot_pose[2]>=(-np.pi*2) and robot_pose[2]<-np.pi:
                theta = robot_pose[2] + (np.pi*2)
            elif robot_pose[2]<=(np.pi*2) and robot_pose[2]>np.pi:
                theta = robot_pose[2] - (np.pi*2)
            else:
                theta = robot_pose[2]
            # print(theta)

            # if x_goal >= x:
            x_diff = np.round_((x_goal - x),8)
            # else:
            #     x_diff = x - x_goal

            # if y_goal >= y:
            y_diff = np.round_((y_goal - y),8)
            # else:
                # y_diff = y - y_goal

            #Obtain angle difference
            alpha = np.arctan2(y_diff, x_diff) - theta

            if alpha>np.pi:
                alpha = alpha - 2*np.pi
            elif alpha < -np.pi:
                alpha = alpha + 2*np.pi
            # print(np.arctan2(y_diff, x_diff))
            # print(theta)
            # print(alpha)
            # print("theta")
            # print(y_diff)
            # print(theta)
            # print(np.arctan2(y_diff, x_diff))
            # print(alpha)
        
            #Obtain length difference
            rho = np.clip(np.hypot(x_diff, y_diff),0,0.6)

            self.turn_time = abs(alpha*(baseline*np.pi/scale/wheel_lr)/(2*np.pi)) # replace with your calculation
            # self.turn_time = (-0.01485446135524356467176248458143*(abs(alpha)-np.pi/4)+1)*self.turn_time
            self.turn_time = (-0.129732395447351627*(abs(alpha)-np.pi/4)+1)*self.turn_time
            self.drive_time = rho * (   (1/scale)/(wheel_lv)   )  # replace with your calculation
            nextwaypoint_bool = False
            if rho < 0.1:
                nextwaypoint_bool = True
                self.lv, self.rv = self.pibot.set_velocity([0, 0])
                self.order =0
                self.Complete_t_d =True
                print('next waypoint')
                time.sleep(2)
            elif self.order ==0 and self.Complete_t_d ==True and nextwaypoint_bool == False: ##turning
                print("Turning for {:.5f} seconds".format(self.turn_time[0]))
                # self.turn_time =0
                if alpha >0:
                    self.lv, self.rv = self.pibot.set_velocity([0, 1], tick=20,turning_tick=wheel_lr, time= self.turn_time[0])
                    self.lv = (0.129732395447351627*(abs(alpha)-np.pi/4)+1)*self.lv*1.126
                    self.rv = (0.129732395447351627*(abs(alpha)-np.pi/4)+1)*self.rv*1.126
                    # print(self.lv, self.rv)
                elif alpha <0:
                    self.lv, self.rv = self.pibot.set_velocity([0, -1], tick=20,turning_tick=wheel_lr,time= self.turn_time[0])
                    self.lv = (0.129732395447351627*(abs(alpha)-np.pi/4)+1)*self.lv*1.126
                    self.rv = (0.129732395447351627*(abs(alpha)-np.pi/4)+1)*self.rv*1.126
                else:
                    self.lv, self.rv = self.pibot.set_velocity([0, 0])
                self.order =1
                self.drivetime_left = None
                self.Complete_t_d =False

                #Keep Track of when code has been executed
                # self.Time_Of_execution_t = time.time()
            # elif self.order ==1 and (time.time() - self.Time_Of_execution_t > self.turn_time[0]) and nextwaypoint_bool == False: ## driving
            elif self.order ==1 and nextwaypoint_bool == False: ## driving
                print("Driving for {:.5f} seconds".format(self.drive_time[0]))
                self.lv, self.rv = self.pibot.set_velocity([1, 0], tick=wheel_lv, time=self.drive_time[0])
                self.order =0
                self.turntime_left = None
                # self.Time_Of_execution_d = time.time()
            # elif self.order ==0 and (time.time()-self.Time_Of_execution_d > self.drive_time[0])and nextwaypoint_bool == False:
            elif self.order ==0 and  nextwaypoint_bool == False:
                self.Complete_t_d =True
                self.lv, self.rv = self.pibot.set_velocity([0, 0])


        time.sleep(1.5)
        ####test
        self.take_pic()
        if not self.data is None:
            self.data.write_keyboard(self.lv, self.rv)

        dt = time.time() - self.control_clock-1.5
        if self.order==0:
            if self.drive_time[0] > dt:
                if self.drivetime_left==None:
                    self.drivetime_left = self.drive_time[0] - dt
                    drive_meas = measure.Drive(self.lv, self.rv, dt)
                elif self.drivetime_left<dt:
                    drive_meas = measure.Drive(self.lv, self.rv, self.drivetime_left)
                    self.drivetime_left ==None
                elif self.drivetime_left!=None:
                    self.drivetime_left = self.drivetime_left-dt
                    drive_meas = measure.Drive(self.lv, self.rv, dt)
            else:
                drive_meas = measure.Drive(self.lv, self.rv, self.drive_time[0])
        else:
            if self.turn_time[0] > dt:
                if self.turntime_left==None:
                    self.turntime_left = self.turn_time[0] - dt
                    drive_meas = measure.Drive(self.lv, self.rv, dt)
                elif self.turntime_left<dt:
                    drive_meas = measure.Drive(self.lv, self.rv, self.turntime_left)
                    self.turntime_left ==None
                elif self.turntime_left!=None:
                    self.turntime_left = self.turntime_left-dt
                    drive_meas = measure.Drive(self.lv, self.rv, dt)
            else:
                drive_meas = measure.Drive(self.lv, self.rv, self.turn_time[0])
        
        # drive_meas = measure.Drive(self.lv, self.rv, dt)
        self.control_clock = time.time()
        operate.update_slam(drive_meas)
            
        return nextwaypoint_bool
    def turn_around(self, robot_pose):
        fileS = "calibration/param/scale.txt"
        scale = np.loadtxt(fileS, delimiter=',') #Distance ticks/m
        fileB = "calibration/param/baseline.txt"
        baseline = np.loadtxt(fileB, delimiter=',') #angle ticks/rad
        # baseline = 1.350086542959292168e-01
        # baseline = 1.300086542959292168e-01
        # wheel_lv = 30 # tick to move the robot
        wheel_lr = 30
        print(robot_pose[0:3])

        if robot_pose[2]>=(np.pi*2):
            theta = robot_pose[2]%(np.pi*2)
            if theta <= (np.pi*2) and theta > (np.pi):
                theta = theta -np.pi*2
        elif robot_pose[2]<(-np.pi*2):
            theta = -(np.pi*2 - robot_pose[2]%(np.pi*2))+(np.pi*2)
            if theta <= (np.pi*2) and theta > (np.pi):
                theta = theta-np.pi*2
        elif robot_pose[2]>=(-np.pi*2) and robot_pose[2]<-np.pi:
            theta = robot_pose[2] + (np.pi*2)
        elif robot_pose[2]<=(np.pi*2) and robot_pose[2]>np.pi:
            theta = robot_pose[2] - (np.pi*2)
        else:
            theta = robot_pose[2]

        if self.around_order ==0 and self.interlock_turn_around == True:
            self.interlock_turn_around = False
            theta_first = theta
            self.points = [theta_first+np.pi/4,theta_first+np.pi/2,theta_first+np.pi*3/4,theta_first+np.pi,theta_first-np.pi*3/4,theta_first-np.pi/2,theta_first-np.pi/4,theta_first]
        
        alpha = self.points[self.around_order] - theta

        if alpha>np.pi:
            alpha = alpha - 2*np.pi
        elif alpha < -np.pi:
            alpha = alpha + 2*np.pi

        self.turn_time_around = abs(alpha*(baseline*np.pi/scale/wheel_lr)/(2*np.pi)) 
        done_ =False
        if abs(alpha)<(5/180*np.pi):
            self.around_order +=1
            self.turntime_left_around==None
            self.lv, self.rv = self.pibot.set_velocity([0, 0])
            if self.around_order >=8:
                self.interlock_turn_around = True
                self.around_order =0
                self.lv, self.rv = self.pibot.set_velocity([0, 0])
                time.sleep(1)
                done_ =True
                print("DONEEEE")
            # print('wwwww')
            # print(self.around_order)
        else:
            print("Turning for {:.5f} seconds".format(self.turn_time_around[0]))
            # self.turn_time =0
            if alpha >0:
                self.lv, self.rv = self.pibot.set_velocity([0, 1], tick=20,turning_tick=wheel_lr, time= self.turn_time_around[0])
                # print(self.lv, self.rv)
            elif alpha <0:
                self.lv, self.rv = self.pibot.set_velocity([0, -1], tick=20,turning_tick=wheel_lr,time= self.turn_time_around[0])
            else:
                self.lv, self.rv = self.pibot.set_velocity([0, 0])
        # print('dddd')
        # print(theta) 
        # print(alpha)
        # # print(self.points[self.around_order])
        # # print(5/180*np.pi)
        # print(done_)
        
        time.sleep(1.5)
        self.take_pic()
        if not self.data is None:
            self.data.write_keyboard(self.lv, self.rv)

        dt = time.time() - self.control_clock-1.5

        if self.turn_time_around[0] > dt:
            if self.turntime_left_around==None:
                self.turntime_left_around = self.turn_time_around[0] - dt
                drive_meas = measure.Drive(self.lv, self.rv, dt)
            elif self.turntime_left_around<dt:
                drive_meas = measure.Drive(self.lv, self.rv, self.turntime_left_around)
                self.turntime_left_around ==None
            elif self.turntime_left_around!=None:
                self.turntime_left_around = self.turntime_left_around-dt
                drive_meas = measure.Drive(self.lv, self.rv, dt)
        else:
            drive_meas = measure.Drive(self.lv, self.rv, self.turn_time_around[0])
        
        # drive_meas = measure.Drive(self.lv, self.rv, dt)
        self.control_clock = time.time()
        operate.update_slam(drive_meas)
            
        return done_
    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            h,  w = self.img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
            # undistort
            self.img = cv2.undistort(self.img, self.mtx, self.dist, None, newcameramtx)

            self.detector_output, self.network_vis = self.detector.yolo_detection(self.img)###########yolov5
            #self.detector_output, self.network_vis = self.detector.detect_single_image(self.img)#####resnet
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'

    # save images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:

            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.network_vis,
                                   (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, 
                                position=(h_pad, 240+2*v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    

    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():
            ########### replace with your M1 codes ###########
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'] = [2.5, 0]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'] = [-2.5, 0]
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'] = [0, 3]
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'] = [0, -3]
            ####################################################
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
                print(self.get_robot_pose()[0:3])
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                #self.command['inference'] = True
                self.CaptureFruit()
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                if(self.nextwaypoint_bool == True and self.not_scanning_around == True):
                    x = input("X coordinate of the waypoint: ")
                    try:
                        x = float(x)
                    except ValueError:
                        print("Please enter a number.")
                        continue
                    y = input("Y coordinate of the waypoint: ")
                    try:
                        y = float(y)
                    except ValueError:
                        print("Please enter a number.")
                        continue
                    self.waypoint = [x,y]
                    self.DrivingCommand = True
                    
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                if(self.nextwaypoint_bool == True and self.not_scanning_around == True):
                    self.TurningCommand = True

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                if(self.aligned == False):
                    # isAruco = True
                    # self.MarkerTransformation()
                    #  = self.MarkerTransformation(isAruco)
                    print("Markers Have Aligned!")
                    self.aligned = True
                else:
                    print("ITS ALREADY ALGINED!")
            
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:

                if len(self.fruits_list) == 5:
                    try:
                        self.target_est['redapple_'+str(0)] = {'x':self.fruits_true_pos[self.numFruitList.index(1)][0], 'y':self.fruits_true_pos[self.numFruitList.index(1)][1]}
                    except:
                        pass
                    try:
                        self.target_est['greenapple_'+str(0)] = {'x':self.fruits_true_pos[self.numFruitList.index(2)][0], 'y':self.fruits_true_pos[self.numFruitList.index(2)][1]}
                    except:
                        pass
                    try:
                        self.target_est['orange_'+str(0)] = {'x':self.fruits_true_pos[self.numFruitList.index(3)][0], 'y':self.fruits_true_pos[self.numFruitList.index(3)][1]}
                    except:
                        pass
                    try:
                        self.target_est['mango_'+str(0)] = {'x':self.fruits_true_pos[self.numFruitList.index(4)][0], 'y':self.fruits_true_pos[self.numFruitList.index(4)][1]}
                    except:
                        pass
                    try:
                        self.target_est['capsicum_'+ str(0)] = {'x':self.fruits_true_pos[self.numFruitList.index(5)][0], 'y':self.fruits_true_pos[self.numFruitList.index(5)][1]}
                    except:
                        pass


                    with open(self.base_dir/'lab_output/targets.txt', 'w') as fo:
                        json.dump(self.target_est, fo)
                    
                    print("Fruits positions Saved!")
                
                else:
                    print("Pics Currently taken: (If less than 3 Take More)")
                    print("Red Apples",len(self.RedapplePics))
                    print("Green Apples",len(self.GreenapplePics))
                    print("Orange",len(self.OrangePics))
                    print("Mango",len(self.MangoPics))
                    print("Capsicum",len(self.CapsicumPics))



        if self.quit:
            pygame.quit()
            sys.exit()

   
    ### MILESTONE 5 CODE
    def get_robot_pose(self):
        return self.ekf.get_state_vector() 
    
    def get_bounding_box(self, target_number, img):
        image = img.resize((640,480), PIL.Image.Resampling.NEAREST)
        target = Image(image)==target_number
        blobs = target.blobs()
        [[u1,u2],[v1,v2]] = blobs[0].bbox # bounding box
        width = abs(u1-u2)
        height = abs(v1-v2)
        center = np.array(blobs[0].centroid).reshape(2,)
        box = [center[0], center[1], int(width), int(height)] # box=[x,y,width,height]
        # plt.imshow(fruit.image)
        # plt.annotate(str(fruit_number), np.array(blobs[0].centroid).reshape(2,))
        # plt.show()
        # assert len(blobs) == 1, "An image should contain only one object of each target type"
        return box

    # read in the list of detection results with bounding boxes and their matching robot pose info
    def get_image_info(self, img, image_poses):
        # there are at most five types of targets in each image
        target_lst_box = [[], [], [], [], []]
        target_lst_pose = [[], [], [], [], []]
        completed_img_dict = {}

        # add the bounding box info of each target in each image
        # target labels: 1 = redapple, 2 = greenapple, 3 = orange, 4 = mango, 5=capsicum, 0 = not_a_target

        
        img_vals = set(Image(img, grey=True).image.reshape(-1))
        for target_num in img_vals:
            if target_num > 0:
                try:
                    box = self.get_bounding_box(target_num, img) # [x,y,width,height]
                    pose = [image_poses[0], image_poses[1], image_poses[2]] # [x, y, theta]
                    target_lst_box[int(target_num-1)].append(box) # bouncing box of target
                    target_lst_pose[int(target_num-1)].append(np.array(pose).reshape(3,)) # robot pose
                except ZeroDivisionError:
                    pass

        # if there are more than one objects of the same type, combine them
        for i in range(5):
            if len(target_lst_box[i])>0:
                box = np.stack(target_lst_box[i], axis=1)
                pose = np.stack(target_lst_pose[i], axis=1)
                completed_img_dict[i+1] = {'target': box, 'robot': pose}
        
        return completed_img_dict

    # estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
    def estimate_pose(self, camera_matrix, completed_img_dict):
        camera_matrix = camera_matrix
        focal_length = camera_matrix[0][0]
        # actual sizes of targets [For the simulation models]
        # You need to replace these values for the real world objects
        target_dimensions = []
        redapple_dimensions = [0.074, 0.074, 0.087]
        target_dimensions.append(redapple_dimensions)
        greenapple_dimensions = [0.081, 0.081, 0.067]
        target_dimensions.append(greenapple_dimensions)
        orange_dimensions = [0.075, 0.075, 0.072]
        target_dimensions.append(orange_dimensions)
        mango_dimensions = [0.113, 0.067, 0.058] # measurements when laying down
        target_dimensions.append(mango_dimensions)
        capsicum_dimensions = [0.073, 0.073, 0.088]
        target_dimensions.append(capsicum_dimensions)

        target_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']

        target_pose_dict = {}
        # for each target in each detection output, estimate its pose
        for target_num in completed_img_dict.keys():
            box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
            robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
            true_height = target_dimensions[target_num-1][2]
            
            if robot_pose[2]>=(np.pi*2):
                theta = robot_pose[2]%(np.pi*2)
                if theta <= (np.pi*2) and theta > (np.pi):
                    theta = theta -np.pi*2
            elif robot_pose[2]<(-np.pi*2):
                theta = -(np.pi*2 - robot_pose[2]%(np.pi*2))+(np.pi*2)
                if theta <= (np.pi*2) and theta > (np.pi):
                    theta = theta-np.pi*2
            elif robot_pose[2]>=(-np.pi*2) and robot_pose[2]<-np.pi:
                theta = robot_pose[2] + (np.pi*2)
            elif robot_pose[2]<=(np.pi*2) and robot_pose[2]>np.pi:
                theta = robot_pose[2] - (np.pi*2)
            else:
                theta = robot_pose[2]

            robot_pose__ = -theta

            # print(box)
            z = true_height*focal_length/box[3]
            x_pose = np.sin(robot_pose__)* (z*(640/2-box[0])/focal_length) + np.cos(robot_pose__)* z + robot_pose[0]
            y_pose = np.cos(robot_pose__)* (z*-(640/2-box[0])/focal_length) - np.sin(robot_pose__)* z + robot_pose[1]

            target_pose = {'x': x_pose, 'y': y_pose}
            
            target_pose_dict[target_list[target_num-1]] = target_pose
            ###########################################

        return target_pose_dict

    def CaptureFruit(self):
            # Take picture of surroundings
            # Find fruits within it
            # If fruit is found that is not on the list execute the following
            # Else Continue
       
        self.command['inference'] = True
        self.take_pic()
        self.detect_target()
        DiscoveredItems = np.unique(self.detector_output)
        print(DiscoveredItems)
        PotentialObstacle = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']
        currentPose = self.get_robot_pose()
        currentPose = [currentPose[0][0], currentPose[1][0], currentPose[2][0]]
        factor = 5
        AreaInterested = (640*480)/(2**factor)

        for i in range(len(DiscoveredItems)):
            if self.numFruitList.count(DiscoveredItems[i]) == 0 and not(DiscoveredItems[i] == 0):

                if DiscoveredItems[i] == 1 and np.count_nonzero(self.detector_output == 1) >= AreaInterested:
                    self.RedapplePics.append(self.detector_output)
                    self.RedapplePose.append(currentPose)
                    print("RedApple Detected")

                elif DiscoveredItems[i] == 2 and np.count_nonzero(self.detector_output == 2) >= AreaInterested:
                    self.GreenapplePics.append(self.detector_output)
                    self.GreenapplePose.append(currentPose)
                    print("Greenapple Detected")
                    
                elif DiscoveredItems[i] == 3 and np.count_nonzero(self.detector_output == 3) >= AreaInterested:
                    self.OrangePics.append(self.detector_output)
                    self.OrangePose.append(currentPose)
                    print("Orange Detected")

                elif DiscoveredItems[i] == 4 and np.count_nonzero(self.detector_output == 4) >= AreaInterested:
                    self.MangoPics.append(self.detector_output)
                    self.MangoPose.append(currentPose)
                    print("Mango Detected")

                elif DiscoveredItems[i] == 5 and np.count_nonzero(self.detector_output == 5) >= AreaInterested:
                    self.CapsicumPics.append(self.detector_output)
                    self.CapsicumPose.append(currentPose)
                    print("Capsicum Detected")
                    
        
        
        if(len(self.RedapplePics)) >= 2 and np.count_nonzero(self.detector_output == 1) >= AreaInterested:

            if not(self.numFruitList.count(1) == 0):
                self.fruits_list.remove("redapple")
                np.delete(self.fruits_true_pos,np.where(self.numFruitList==1))
                np.delete(self.numFruitList, np.where(self.numFruitList==1))
            
            self.numFruitList.append(1)
            self.fruits_list.append(PotentialObstacle[0])
            self.EstimationAndMerge(self.RedapplePics, self.RedapplePose, 0, self.RedappleFound)
            self.RedappleFound = True
        
        if(len(self.GreenapplePics)) >= 2 and np.count_nonzero(self.detector_output == 2) >= AreaInterested:
            if not(self.numFruitList.count(2) == 0):
                self.fruits_list.remove("greenapple")
                np.delete(self.fruits_true_pos,np.where(self.numFruitList==2))
                np.delete(self.numFruitList, np.where(self.numFruitList==2))

            self.numFruitList.append(2) 
            self.fruits_list.append(PotentialObstacle[1])
            self.EstimationAndMerge(self.GreenapplePics, self.GreenapplePose, 1, self.GreenappleFound)
            self.GreenappleFound = True

        if(len(self.OrangePics) >= 2 and np.count_nonzero(self.detector_output == 3) >= AreaInterested) :
            if not(self.numFruitList.count(3) == 0):
                self.fruits_list.remove("orange")
                np.delete(self.fruits_true_pos,np.where(self.numFruitList==3))
                np.delete(self.numFruitList, np.where(self.numFruitList==3))

            self.numFruitList.append(3)
            self.fruits_list.append(PotentialObstacle[2])
            self.EstimationAndMerge(self.OrangePics, self.OrangePose, 2,self.OrangeFound)
            self.OrangeFound = True
     
        if(len(self.MangoPics) >= 2) and np.count_nonzero(self.detector_output == 4) >= AreaInterested:
            if not(self.numFruitList.count(4) == 0):
                self.fruits_list.remove("mango")
                np.delete(self.fruits_true_pos,np.where(self.numFruitList==4))
                np.delete(self.numFruitList, np.where(self.numFruitList==4))

            self.numFruitList.append(4)
            self.fruits_list.append(PotentialObstacle[3])
            self.EstimationAndMerge(self.MangoPics, self.MangoPose, 3, self.MangoFound)
            self.MangoFound = True
           

        if(len(self.CapsicumPics) >= 2) and np.count_nonzero(self.detector_output == 5) >= AreaInterested:
            if not(self.numFruitList.count(5) == 0):
                self.fruits_list.remove("capsicum")
                np.delete(self.fruits_true_pos,np.where(self.numFruitList==5))
                np.delete(self.numFruitList, np.where(self.numFruitList==5))

            self.numFruitList.append(5)
            self.fruits_list.append(PotentialObstacle[4])
            self.EstimationAndMerge(self.CapsicumPics, self.CapsicumPose, 4, self.CapsicumFound)
            self.CapsicumFound = True
            

        
        
            # Estimate position
            # Use Target Pose Est
            # Set regeneration to true
            # Mark detected fruit as known fruit
    def EstimationAndMerge (self, pictureArray, poseArray, num, Found):

        # picture array - detector output
        # PoseArray - array of poses of robot
        # num - index of fruit in possible obstacle
        PotentialObstacle = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']
        xarray = []
        yarray = []

        for i in range(len(pictureArray)):
            CurrentImage = PIL.Image.fromarray(pictureArray[i])
            completed_img_dict = self.get_image_info(CurrentImage, poseArray[i])
            target_map = self.estimate_pose(self.camera_matrix, completed_img_dict)
            xarray.append(target_map[PotentialObstacle[num]]['x'])
            yarray.append(target_map[PotentialObstacle[num]]['y'])


        est_x = statistics.median(xarray)
        est_y = statistics.median(yarray)



        
        if self.fruits_true_pos[0][0] == 0 and self.fruits_true_pos[0][1] == 0:
            self.fruits_true_pos = np.array([est_x[0],est_y[0]]).reshape(1,2)
            print("1")
            print("Unknown Fruit Detected!!!")
            print(PotentialObstacle[num])
            print(np.array([est_x[0],est_y[0]]))

        elif Found == False:
            print("2")
            self.fruits_true_pos = np.append(self.fruits_true_pos,np.array([est_x[0],est_y[0]]).reshape(1,2), axis=0)
            print("Unknown Fruit Detected!!!")
            print(PotentialObstacle[num])
            print(np.array([est_x[0],est_y[0]]))

        else:
            print("3")
            self.fruits_true_pos = np.append(self.fruits_true_pos,np.array([est_x[0],est_y[0]]).reshape(1,2), axis=0)
            print("New Position")
            print(PotentialObstacle[num])
            print(np.array([est_x[0],est_y[0]]))

        print("New Fruit list")
        print(self.fruits_true_pos)
        print(self.fruits_list)

    # def MarkerTransformation(self):
    #     #Define robot current position
    #     #Find transformation frame from robot to the real world
    #     #Apply Transformation matrix onto all frames
    #     angle = self.get_robot_pose()[2]
    #     robot_xy_translate = self.get_robot_pose()[0:2,:]
    #     # print(angle)
    #     # print(robot_xy_translate)
    #     # print(isAruco)
    #     xArray = []
    #     yArray = []
        
    #     c, s = np.cos(-angle), np.sin(-angle)
    #     R = np.array(((c, -s), (s, c))).reshape(2,2)

    #     # print("Begin Test")
    #     # print(self.ekf.taglist)
    #     # print(angle)
    #     # print(x_translate)
    #     # print(y_translate)
    #     # if isAruco == True:
    #     for i in range(len(self.ekf.taglist)):
    #         coordinateArray = np.zeros((2,1))

            
    #         coordinateArray[0,0] = self.ekf.markers[0][i]
    #         coordinateArray[1,0] = self.ekf.markers[1][i]

    #         TransformedMatrix = R @ coordinateArray - robot_xy_translate

    #         newX = TransformedMatrix[0,0]
    #         newY = TransformedMatrix[1,0]
    #         # print(coordinateArray)
    #         # print(self.ekf.taglist[i])
    #         # print(newX)
    #         # print(newY)
    #         xArray = np.append(xArray, newX)
    #         yArray = np.append(yArray, newY)

    #     self.ekf.markers = np.array([xArray,yArray])

    #     # else:
    #     for i in range(len(self.fruits_list)):
    #         coordinateArray = np.zeros((2,1))

            
    #         coordinateArray[0,0] = self.fruits_true_pos[i][0]
    #         coordinateArray[1,0] = self.fruits_true_pos[i][0]

    #         TransformedMatrix = R @ coordinateArray - robot_xy_translate

    #         newX = TransformedMatrix[0,0]
    #         newY = TransformedMatrix[1,0]
    #         # print(coordinateArray)
    #         # print(self.ekf.taglist[i])
    #         # print(newX)
    #         # print(newY)
    #         self.fruits_true_pos[i] = [newX, newY]
    def MarkerTransformation(self):
        #Define robot current position
        #Find transformation frame from robot to the real world
        #Apply Transformation matrix onto all frames
        robot_pose = self.get_robot_pose()

        if robot_pose[2]>=(np.pi*2):
            theta = robot_pose[2]%(np.pi*2)
            if theta <= (np.pi*2) and theta > (np.pi):
                theta = theta -np.pi*2
        elif robot_pose[2]<(-np.pi*2):
            theta = -(np.pi*2 - robot_pose[2]%(np.pi*2))+(np.pi*2)
            if theta <= (np.pi*2) and theta > (np.pi):
                theta = theta-np.pi*2
        elif robot_pose[2]>=(-np.pi*2) and robot_pose[2]<-np.pi:
            theta = robot_pose[2] + (np.pi*2)
        elif robot_pose[2]<=(np.pi*2) and robot_pose[2]>np.pi:
            theta = robot_pose[2] - (np.pi*2)
        else:
            theta = robot_pose[2]

        x = np.array([robot_pose[0], robot_pose[1]])
        
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array(((c, -s), (s, c))).reshape(2,2)

        if len(self.ekf.taglist)>0:
            yArray = [self.ekf.markers[0][0]]
            xArray = [self.ekf.markers[0][1]]
            
            for i in range(len(self.ekf.taglist)-1):
                xArray = np.append(xArray, self.ekf.markers[0][i+1])
                yArray = np.append(yArray, self.ekf.markers[1][i+1])
            
            self.ekf.markers = np.array([xArray,yArray])
            self.ekf.markers = R @ self.ekf.markers - x
        
        
        if len(self.fruits_list)>0:
            yArray = [self.fruits_true_pos[0][0]]
            xArray = [self.fruits_true_pos[0][1]]
            
            for i in range(len(self.fruits_list)-1):
                xArray = np.append(xArray, self.fruits_true_pos[i+1][0])
                yArray = np.append(yArray, self.fruits_true_pos[i+1][1])

            Array = np.array([xArray,yArray])
            Array = R @ Array - x
            
            for i in range(len(self.fruits_list)):
                self.fruits_true_pos[i] = [Array[0][i], Array[1][i]]


        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    args, _ = parser.parse_known_args()
    
    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()
    operate = Operate(args)

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    fileK = "{}intrinsic.txt".format('./calibration/param/')
    # filedis = "{}distCoeffs.txt".format('./calibration/param/')
    operate.camera_matrix = np.loadtxt(fileK, delimiter=',')

    while start:
        operate.update_keyboard()
        operate.take_pic()
        
        if(operate.DrivingCommand):
            operate.nextwaypoint_bool = operate.control(operate.waypoint,operate.get_robot_pose())
            if(operate.nextwaypoint_bool == True):
                operate.DrivingCommand = False
        
        elif(operate.TurningCommand):
            
            operate.not_scanning_around = operate.turn_around(operate.get_robot_pose())
            if(operate.not_scanning_around == True):
                operate.TurningCommand = False
        
        else:
            drive_meas = operate.Stationarycontrol()
            operate.update_slam(drive_meas)


        operate.record_data()
        operate.save_image()

        
        # visualise
        operate.draw(canvas)
        pygame.display.update()
