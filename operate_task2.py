# teleoperate the robot, perform SLAM and object detection

# basic python packages
import numpy as np
import cv2 
import os, sys
import time
import json
import ast

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import Alphabot # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector


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
        self.ekf.init_lm_cov = 0
        self.ekf.R_Factor = 0 
        self.ekf.Q_Factor = 0
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.06) # size of the ARUCO markers

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
        self.count_down = 300
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
        ### Task1 ###
        self.order =0
        self.Complete_t_d =True
        self.lv, self.rv = [0,0]
        self.drive_time =[0]
        self.turn_time =[0]
        self.turntime_left_around =None
        self.turntime_left1 = None
        self.turntime_left2 = None
        self.drivetime_left = None
        ### Task2 ###
        self.fruits_true_pos = []
        self.aruco_true_pos=[]
        self.start = False
        self.around_order =0
        self.interlock_turn_around = True
        self.points =[]
        self.TruePositions = []

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
            # self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            h,  w = self.img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
            # undistort
            self.img = cv2.undistort(self.img, self.mtx, self.dist, None, newcameramtx)
            
            self.detector_output, self.network_vis = self.detector.detect_single_image(self.img)
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'

    # save raw images taken by the camera
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
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                        self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

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
            # if self.start:
            #     start = True
            # else: 
            #     start = False
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
                self.start = True
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    # if not self.ekf_on:
                        # self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()


    ############ Milestone 4 ###################
    def read_true_map_fruit(self, fname):
            """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search
            @param fname: filename of the map
            @return:
                1) list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
                2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
                3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
            """
            with open(fname, 'r') as f:
                try:
                    gt_dict = json.load(f)                   
                except ValueError as e:
                    with open(fname, 'r') as f:
                        gt_dict = ast.literal_eval(f.readline())   
                fruit_list = []
                fruit_true_pos = []
                # aruco_true_pos = np.empty([10, 2])

                # remove unique id of targets of the same type
                for key in gt_dict:
                    x = np.round(gt_dict[key]['x'], 1)
                    y = np.round(gt_dict[key]['y'], 1)

                    # if key.startswith('aruco'):
                    #     if key.startswith('aruco10'):
                    #         aruco_true_pos[9][0] = x
                    #         aruco_true_pos[9][1] = y
                    #     else:
                    #         marker_id = int(key[5])
                    #         # print(marker_id)
                    #         aruco_true_pos[marker_id-1][0] = x
                    #         aruco_true_pos[marker_id-1][1] = y
                    # else:
                    fruit_list.append(key[:-2])
                    if len(fruit_true_pos) == 0:
                        fruit_true_pos = np.array([[x, y]])
                    else:
                        fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

                return fruit_list, fruit_true_pos#, aruco_true_pos
    def read_true_map_slam(self, fname):
            with open(fname, 'r') as f:
                try:
                    gt_dict = json.load(f)                   
                except ValueError as e:
                    with open(fname, 'r') as f:
                        gt_dict = ast.literal_eval(f.readline())   
                aruco_true_pos = np.empty([10, 2])

                for i in range(len(gt_dict['taglist'])):
                    if gt_dict['taglist'][i]>=1 and gt_dict['taglist'][i]<=10:
                        x = np.round(gt_dict['map'][0][i], 1)
                        y = np.round(gt_dict['map'][1][i], 1)
                        
                        aruco_true_pos[(int(gt_dict['taglist'][i])-1)][0] = x
                        aruco_true_pos[(int(gt_dict['taglist'][i])-1)][1] = y
                return aruco_true_pos

    def read_search_list(self):
        """Read the search order of the target fruits

        @return: search order of the target fruits
        """
        search_list = []
        with open('search_list.txt', 'r') as fd:
            fruits = fd.readlines()

            for fruit in fruits:
                search_list.append(fruit.strip())

        return search_list

    def print_target_fruits_pos(self, search_list, fruit_list, fruit_true_pos):
        """Print out the target fruits' pos in the search order

        @param search_list: search order of the fruits
        @param fruit_list: list of target fruits
        @param fruit_true_pos: positions of the target fruits
        """
        self.fruit_orderlist = []
        print("Search order:")
        n_fruit = 1
        for fruit in search_list:
            for i in range(3):
                if fruit == fruit_list[i]:
                    print('{}) {} at [{}, {}]'.format(n_fruit,
                                                      fruit,
                                                      np.round(fruit_true_pos[i][0], 1),
                                                      np.round(fruit_true_pos[i][1], 1)))
                    if len(self.fruit_orderlist) == 0:
                        self.fruit_orderlist = np.array([[fruit,fruit_true_pos[i][0],fruit_true_pos[i][1]]])
                    else:
                        self.fruit_orderlist = np.append(self.fruit_orderlist, [[fruit,fruit_true_pos[i][0],fruit_true_pos[i][1]]], axis=0)
            n_fruit += 1

    def get_robot_pose(self):
        return self.ekf.get_state_vector() 

    ################## Task 1 #############################
    # wheel control

    def innitMarkers(self, Positions):
        for i in range(len(Positions)):
            x = Positions[i][0]
            y = Positions[i][1]
            marker = np.block([[x],[y]]).reshape(-1,1)

            landmarks = measure.Marker(marker, i+1)
            self.TruePositions.append(landmarks)
        self.ekf.add_landmarks(self.TruePositions)




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
            print(robot_pose)

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
        
            #Obtain length difference
            rho = np.clip(np.hypot(x_diff, y_diff),0,0.8)

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
                # time.sleep(0.5)
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
            # elif self.order ==0.5:
            #     self.order =1
            elif self.order ==1 and nextwaypoint_bool == False: #second turning for fine tune
                print("Second turning for {:.5f} seconds".format(self.turn_time[0]))
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
                self.order =2
                self.turntime_left1 = None
                # self.drivetime_left = None
                # self.Complete_t_d =False

            elif self.order ==2 and nextwaypoint_bool == False: ## driving
                print("Driving for {:.5f} seconds".format(self.drive_time[0]))
                self.lv, self.rv = self.pibot.set_velocity([1, 0], tick=wheel_lv, time=self.drive_time[0])
                self.order =0
                self.turntime_left2 = None
                # self.Time_Of_execution_d = time.time()
            # elif self.order ==0 and (time.time()-self.Time_Of_execution_d > self.drive_time[0])and nextwaypoint_bool == False:
            elif self.order ==0 and  nextwaypoint_bool == False:
                self.Complete_t_d =True
                self.lv, self.rv = self.pibot.set_velocity([0, 0])


        time.sleep(1)
        ####test
        self.take_pic()
        if not self.data is None:
            self.data.write_keyboard(self.lv, self.rv)

        dt = time.time() - self.control_clock-1
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
        elif self.order ==1:
            print('turn1')
            if self.turn_time[0] > dt:
                if self.turntime_left1==None:
                    self.turntime_left1 = self.turn_time[0] - dt
                    drive_meas = measure.Drive(self.lv, self.rv, dt)
                elif self.turntime_left1<dt:
                    drive_meas = measure.Drive(self.lv, self.rv, self.turntime_left1)
                    self.turntime_left1 ==None
                elif self.turntime_left1!=None:
                    self.turntime_left1 = self.turntime_left1-dt
                    drive_meas = measure.Drive(self.lv, self.rv, dt)
            else:
                drive_meas = measure.Drive(self.lv, self.rv, self.turn_time[0])
        else:
            print('turn2')
            if self.turn_time[0] > dt:
                if self.turntime_left2==None:
                    self.turntime_left2 = self.turn_time[0] - dt
                    drive_meas = measure.Drive(self.lv, self.rv, dt)
                elif self.turntime_left2<dt:
                    drive_meas = measure.Drive(self.lv, self.rv, self.turntime_left2)
                    self.turntime_left2 ==None
                elif self.turntime_left2!=None:
                    self.turntime_left2 = self.turntime_left2-dt
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
        print(robot_pose)

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
        
        time.sleep(1)
        self.take_pic()
        if not self.data is None:
            self.data.write_keyboard(self.lv, self.rv)

        dt = time.time() - self.control_clock-1

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
    ################Task 2########################
    def obstacles_detector(self,node):##return True if obstacles is near (added by zi yu)
        for count,(x,y) in enumerate(self.aruco_true_pos):
            if count!=0:
                x_diff = node[0] - x
                y_diff = node[1] - y
                if np.hypot(x_diff, y_diff)<0.15:
                    return True
        
        for x,y in self.fruits_true_pos:
            x_diff = node[0] - x
            y_diff = node[1] - y
            if np.hypot(x_diff, y_diff)<0.15:
                return True
        return False
    
    def displacement_from_target(self,node,target): ##return dispalcement of target and node (added by zi yu)
        x_diff = node[0] - float(target[1])
        y_diff = node[1] -float(target[2])
        return np.hypot(x_diff, y_diff)

    def neighbor_nodes(self,robot_posnode):
        
        nodes = []
        for i in range(9):
            if i ==0:
                x____ = robot_posnode[0] -0.2
                y____ = robot_posnode[1] +0.2
                nodes = np.array([[x____,y____]])
            else:
                x____ += 0.2
                x____ = np.round_(x____, decimals=1)
                if (i == 3 or i ==6) and i!=0:
                    y____ -= 0.2
                    y____ = np.round_(y____, decimals=1)
                    x____ = robot_posnode[0] -0.2
                if i!=4:
                    if len(nodes)==0:
                        nodes = np.array([[x____,y____]])
                    else:
                        nodes = np.append(nodes,[[x____,y____]],axis=0)
        return nodes

    def optimum_nodes_to_target(self,neighbor_nodes,target):
        optimum_nodes = []
        dis = 999999
        index_action =9
        for index,node in enumerate(neighbor_nodes):
            dis_node_to_target = self.displacement_from_target(node,target)
            if (index == 0 or index ==2 or index ==5 or index ==7) and (dis>(dis_node_to_target+0.09)) and not (self.obstacles_detector(node)):
                optimum_nodes = node
                dis = dis_node_to_target
                index_action =index
            elif not (index == 0 or index ==2 or index ==5 or index ==7) and dis>dis_node_to_target and not (self.obstacles_detector(node)):
                optimum_nodes = node
                dis = dis_node_to_target
                index_action =index
        return optimum_nodes,index_action

    def optimum_waypoint_from_robotpos(self,robot_posnode,target): 
        completesearch_waypoint =False
        # previous_pos = np.reshape(robot_posnode, (-1))
        neighbor_nodes = self.neighbor_nodes(robot_posnode)
        neighbor_nodes = np.reshape(neighbor_nodes, (-1, 2))
        optimum_nodes,index = self.optimum_nodes_to_target(neighbor_nodes,target)
        optimum_nodes = np.reshape(optimum_nodes, (-1, 2))
        waypoints =np.array(optimum_nodes)
        index_action =np.array([[index]])

        while not (completesearch_waypoint):
            neighbor_nodes = self.neighbor_nodes(waypoints[-1,:])
            neighbor_nodes = np.reshape(neighbor_nodes, (-1, 2))
            optimum_nodes,index = self.optimum_nodes_to_target(neighbor_nodes,target)
            optimum_nodes = np.reshape(optimum_nodes, (-1, 2))

            # if index_action[-1] ==index or np.dot([optimum_nodes[-1,0]-waypoints[-1,0],optimum_nodes[-1,1]-waypoints[-1,1]],[waypoints[-1,0]-previous_pos[0],waypoints[-1,1]-previous_pos[1]])==0:
            if index_action[-1] ==index:
                waypoints[-1]=optimum_nodes
                index_action[-1] = index
            else:
                waypoints = np.append(waypoints,optimum_nodes,axis=0)
                index_action = np.append(index_action,[[index]],axis=0)

            # if len(waypoints)>1:
            #     previous_pos = np.reshape(waypoints[-2], (-1))
                
            if self.displacement_from_target(waypoints[-1,:],target)<0.3:
                completesearch_waypoint = True
        return waypoints



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--fruitmap", type=str, default='lab_output/targets.txt')
    parser.add_argument("--slammap", type=str, default='lab_output/slam.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/best.pth')
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

    fruits_list, operate.fruits_true_pos = operate.read_true_map_fruit(args.fruitmap)
    operate.aruco_true_pos = operate.read_true_map_slam(args.slammap)

    search_list = operate.read_search_list()
    operate.print_target_fruits_pos(search_list, fruits_list, operate.fruits_true_pos)

    # waypoints = [[0.8,0],[0.8,0.4]]
    reached_target = None
    target_num =0
    waypoints_num = 0
    not_scanning_around = False
    interlock_waypoints_num_3 = False

    operate.innitMarkers(operate.aruco_true_pos)
    print(fruits_list)
    while start:
        operate.update_keyboard()
        
        if operate.start:
            # operate.take_pic()
            if (waypoints_num%3 ==0 and waypoints_num!=0 and interlock_waypoints_num_3 == False) or reached_target == True or not_scanning_around == False:
                not_scanning_around = operate.turn_around(operate.get_robot_pose())
                interlock_waypoints_num_3 = True
                if not_scanning_around==True and waypoints_num==0 :
                    reached_target = None
                # print('dddd')
            else:
                if reached_target ==True or reached_target ==None:
                    print('fruit target')
                    print(operate.fruit_orderlist[target_num])
                    wayspoints = operate.optimum_waypoint_from_robotpos(operate.get_robot_pose(),operate.fruit_orderlist[target_num])
                    print('waypoints')
                    print(wayspoints)
                    reached_target =False
                
                nextwaypoint_bool = operate.control(wayspoints[waypoints_num],operate.get_robot_pose())

                if nextwaypoint_bool == True:
                    waypoints_num +=1
                    interlock_waypoints_num_3=False
                    if waypoints_num >= len(wayspoints):
                        waypoints_num=0
                        reached_target = True
                        not_scanning_around = False
                        target_num+=1

                        
                        if target_num>= len(search_list):
                            operate.lv, operate.rv = operate.pibot.set_velocity([0, 0])
                            print('DONE moving to all fruits')
                            time.sleep(5)
                            pygame.quit()
                            sys.exit()

                        time.sleep(5)
                        # wait = True
                        # print('press Enter to continue next fruit')
                        # while wait:
                        #     for event in pygame.event.get():
                        #         if event.type == pygame.KEYDOWN:
                        #             wait = False
                    print('wayspoints to')
                    print(wayspoints[waypoints_num])
                
            # time.sleep(7)
        
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()




