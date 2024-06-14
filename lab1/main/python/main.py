#!/usr/bin/env python3

# builtins
import os
import sys
import copy
import math
from time import sleep
import time
from statistics import median, stdev
from math import pi, tau, dist, fabs, cos
import numbers
import random
from project_tools import generate_fake_continous_noise

# packages
import cv2
import geometry_msgs.msg
import moveit_commander
import numpy
import numpy as np
import rospy
import std_msgs
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TwistStamped
from moveit_msgs.msg import DisplayTrajectory
from sensor_msgs.msg import CompressedImage, Image
from blissful_basics import singleton, LazyDict, Warnings
from matplotlib import pyplot as plt
import face_recognition

# Warnings.disable() # uncomment if you want to disable all warnings

sys.path.append(os.path.dirname(__file__))
from project_tools import JointPositions, time_since_prev, clip_value, convert_args
from project_tools import send_to_survivor_bud, generate_fake_continous_noise

# NOTE: you can edit anything, so if you don't like the structure just change it

config = LazyDict(
    send_to_rviz=True,
    video_width=640,
    video_height=480,
    
    BLAH_BLAH_BLAH_YOUR_VARIABLE_HERE="something",
    # NOTE: running python ./main/python/main.py --BLAH_BLAH_BLAH_YOUR_VARIABLE_HERE 99
    #       will effectively change the "something" to 99
)

# Defining Schemas
# Behavior Schema
class BehaviorSchema:
    def __init__(self, perceptualScheme, motorScheme):
        self.perceptualSchema = perceptualScheme
        self.motorSchema = motorScheme
        self.timeSinceAction = 0
        self.timeInAction = 0
        self.currentAction = None
        self.state = {"currentState": "neutral", "time": time.time()}
        self.audioData = None
        self.visualData = None
        self.previousFace = None
        self.timeSinceFace = 0  # This is bc face detection is finnicky

    # Performing behavior function
    def performBehavior(self):
        # Determining if a loud noise has been detected
        loudNoise, intensity = self.perceptualSchema.isLoudNoise(self.audioData)
        print("Current state: ", self.state["currentState"])
        # No matter what, if loud noise, perform the fearful action
        if loudNoise:
            # Updating state
            self.state["currentState"] = "fearful"
            self.state["time"] = time.time()

            # Performing the fearfulAction
            self.motorSchema.fearfulAction()

        # Determining if moderate noise
        moderateNoise = self.perceptualSchema.isModerateNoise(self.audioData)
        if moderateNoise:
            # Starting the speaking
            self.motorSchema.talk()

        # Determing faces, face distance and location
        if self.visualData is not None:
            isFace, isFaceClose, faceLocation = self.perceptualSchema.detectFace(self.visualData)
        else:
            isFace = False
            isFaceClose = False
            faceLocation = 0

        if self.state["currentState"] != "fearful":  # Can't interrupt fearful action
            # If a face is detected
            if isFace:
                self.timeSinceFace = time.time()
                self.previousFace = faceLocation
                self.state["time"] = time.time() # Updating time
                # If the currentState is startled or happy, and 2 seconds has elapsed, start tracking
                if (self.state["currentState"] == "startled" and isFaceClose):
                    self.motorSchema.startledTrack(faceLocation * -1)
                    
                elif (self.state["currentState"] == "happy" and not isFaceClose):
                    self.motorSchema.happyTrack(faceLocation * -1)
                
                # Next two are for changing between the two states
                elif (self.state["currentState"] == "startled" and not isFaceClose):
                    self.state["currentState"] = "happy"
                    self.motorSchema.happyAction()
                
                elif (self.state["currentState"] == "happy" and isFaceClose):
                    self.state["currentState"] = "startled"
                    self.motorSchema.startledAction()

                # If the state is neutral
                elif self.state["currentState"] == "neutral":
                    # If isFaceClose, then we need to do startled
                    if isFaceClose:
                        self.state["currentState"] = "startled"
                        self.motorSchema.startledAction()

                    else:  # Not too close
                        self.state["currentState"] = "happy"
                        self.motorSchema.happyAction()


            # If face no longer in the frame, determine if it was in the middle and disappeared
            else:
                if time.time() - self.timeSinceFace > 3:
                    if self.previousFace is not None:
                        if -0.3 < self.previousFace < 0.3 and self.state["currentState"] != "curious":
                            self.state["currentState"] = "curious"
                            self.state["time"] = time.time()
                            self.motorSchema.curiousAction()
                            self.previousFace = None
                        
        # Checking for if returning to neutral
        if time.time() - self.state["time"] >= 5 and self.state["currentState"] != "neutral":
            # Updating state
            self.state["currentState"] = "neutral"
            self.state["time"] = time.time()

            # Performing neutral action
            self.motorSchema.neutralAction()
            self.previousFace = None


# Perceptual Schema
class ClapPerceptual:
    def __init__(self, sensitivity, loud_threshold):
        self.audioData = None
        self.visualData = None
        self.sensitivity = sensitivity
        self.loud_threshold = loud_threshold
        self.recent_values = []
        self.loud_events = []  # Timestamps of loud events
        
    def update_recent_volumes(self, volume):
        # If the recent values is at 100, remove earliest value
        if len(self.recent_values) < 50:
            self.recent_values.append(volume)

    def isLoudNoise(self, audio_data):
        # Simple volume spike detection as a placeholder for clapping detection
        volume = np.mean(np.abs(audio_data))
        self.update_recent_volumes(volume)

        # Determining if there is enough data in the volumes list
        if len(self.recent_values) >= 5:
            average_volume = np.mean(self.recent_values[:-5])
            threshold = average_volume * self.sensitivity
            # Returns true if above the threshold
            if volume >= 0.6:
                
                # Adding to loud timestamps
                self.loud_events.append(time.time())
                return True, volume
            else:
                return False, 0
            
        else:
            return False, 0
        
    # Moderate noise
    def isModerateNoise(self, audio_data):
        volume = np.mean(np.abs(audio_data))
        self.update_recent_volumes(volume)
        
        if len(self.recent_values) >= 5:
            if volume >= 0.4:
                return True
            
        return False
        
    def detectFace(self, visualData):  # Want to return isFace, isFaceClose, location
        # Converting visualData to RGB
        rgb_frame = (visualData[:, :, ::-1])
        # Detecting face
        face_landmarks_list = face_recognition.face_locations(rgb_frame)

        # TODO: Implement the tracking of the face using the location of the face on the frame
        
        # If there is a face detected
        if len(face_landmarks_list) > 0:
            top = face_landmarks_list[0][0]
            right = face_landmarks_list[0][1]
            bottom = face_landmarks_list[0][2]
            left = face_landmarks_list[0][3]
            
            # Calculating area to determine "distance"
            area = (right - left) * (bottom - top)
            
            # Calculating the location normalized to -1 to 1
            x_loc_absolute = ((right - left) / 2) + left
            x_loc_relative = 2 * ((x_loc_absolute) / rgb_frame.shape[1]) - 1
            
            y_loc_absolute = ((bottom - top) / 2) + top
            y_loc_relative = 2 * ((y_loc_absolute) / rgb_frame.shape[0]) - 1
            
            location = x_loc_relative
            print("Area of face: ", area)
            # If area over an arbitrary threshold, it is too close, if not, return false
            if area > 30000:
                return True, True, location
            else:
                return True, False, location
        
        # If no face, return false and area of 0
        return False, False, 0
    
    def isMiddleGone(self):
        return False
        

# Motor Schema
class ClapMotor:
    def __init__(self, action):
        self.action = action

    def curiousAction(self):
        # Set the joint positions
        Robot.move_towards_positions(
            JointPositions(
                torso_joint=0,
                neck_swivel=45,
                head_tilt=45,
                head_nod=-20
            )
        )

        # Tell robot to not be fearful
        Robot.tell_camera_server("curious")
            #         JointPositions(
        #             torso_joint=torso_joint, # NOTE: units = degrees
        #             neck_swivel=neck_swivel, # <- more negative means more to your left side (the survivor buddy's right side)
        #             head_tilt=head_tilt, # idk which way is which, but tilt is the "roll" in "yaw, pitch, roll"
        #             head_nod=-head_nod, # <- more negative = face the cieling
        #         )
        
    def fearfulAction(self):
        # TODO: Adjust these movements
        Robot.move_towards_positions(
            JointPositions(
                torso_joint=0,
                neck_swivel=-40,
                head_tilt=0,
                head_nod=-20
            )
        )

        # Send the command to the websocket for fearful face
        Robot.tell_camera_server("fearful")
        

    def startledAction(self):
        Robot.move_towards_positions(
            JointPositions(
                torso_joint=-25,
                neck_swivel=0,
                head_tilt=0,
                head_nod=-25
            )
        )

        # Send command to not be fearful
        Robot.tell_camera_server("startled")

    def startledTrack(self, location):
        # TODO implement location tracking
        currentSwivel = Robot.previous_joint_positions.neck_swivel


        print("Tracking")
        if location < 0:
            Robot.move_towards_positions(
                JointPositions(
                    torso_joint=-25,
                    neck_swivel=currentSwivel - 10,
                    head_tilt=0,
                    head_nod=-25
                )
            )
        else:
            Robot.move_towards_positions(
                JointPositions(
                    torso_joint=-25,
                    neck_swivel=currentSwivel + 10,
                    head_tilt=0,
                    head_nod=-25
                )
            )
    
    def happyAction(self):
        # Tell camera server happy
        Robot.tell_camera_server("happy")
        Robot.move_towards_positions(
            JointPositions(
                torso_joint=0,
                neck_swivel=0,
                head_tilt=0,
                head_nod=0
            )
        )

    def happyTrack(self, location):
        # Getting current location
        currentSwivel = Robot.previous_joint_positions.neck_swivel
        currentTorse = Robot.previous_joint_positions.torso_joint
        
        # Determining which direction to move
        print("tracking")
        if location < 0:
            Robot.move_towards_positions(
                JointPositions(
                    torso_joint=0,
                    neck_swivel=currentSwivel - 10,
                    head_tilt=0,
                    head_nod=0
                )
            )
        else:
            Robot.move_towards_positions(
                JointPositions(
                    torso_joint=0,
                    neck_swivel=currentSwivel + 10,
                    head_tilt=0,
                    head_nod=0
                )
            )

    def neutralAction(self):
        Robot.move_towards_positions(
            JointPositions(
                torso_joint=0,
                neck_swivel=0,
                head_tilt=0,
                head_nod=0
            )
        )

        # Tell robot server to return to starting
        Robot.tell_camera_server("returning")

    def talk(self):
        # Telling robot camera server to talk
        Robot.tell_camera_server("talk")



# Creating behavior object with the perception and motor objects
clapPerception = ClapPerceptual(10, 5)
clapMotor = ClapMotor("curious")
clapBehavior = BehaviorSchema(clapPerception, clapMotor)
print(time.time())

class Robot:
    status = LazyDict(
        frame_count=0,
        has_initialized=False,
        # EDIT ME, add stuff to your robot status
        # (you dont have to, but it should be helpful)
    )
    
    def when_audio_chunk_received(chunk):
        data = numpy.array(chunk.data)
        data = generate_fake_continous_noise(delay=2, duration=20, noise_volume=1)

        # Updating audio data and performing behavior
        clapBehavior.audioData = data
        clapBehavior.performBehavior()
            
        # print(f'''Audio data chunk shape is: {data.shape}''')
        # NOTE: Units are unknown (try plotting the data)
        
        # 
        # Edit me (example code)
        # 
        # if True:
        #     # print(f'''Howdy!''')
        #     # print(f'''      config.BLAH_BLAH_BLAH_YOUR_VARIABLE_HERE = {config.BLAH_BLAH_BLAH_YOUR_VARIABLE_HERE}''')
        #     # print(f'''      config["BLAH_BLAH_BLAH_YOUR_VARIABLE_HERE"] = {config["BLAH_BLAH_BLAH_YOUR_VARIABLE_HERE"]}''')
        #     # print(f'''      config["example_arg"] = {config.get("example_arg",None)}''')
        #     # print(f'''Some info:''')
        #     # print(f'''    Robot.previous_joint_positions are:\n{Robot.previous_joint_positions}''')
        #     # print(f'''sleeping for a moment''')
        #     # sleep(1)
        #     # print(f'''moving robot joints''')
        #     torso_joint = 5
        #     neck_swivel = 5
        #     head_tilt = 5
        #     head_nod = 5

        #     Robot.move_towards_positions(
        #         # Note: JointPositions is just a wrapper for these 4 values
        #         #       it tries to keep the numbers within-bounds, but thats about it 
        #         #       Example:
        #         #           JointPositions([0,1,2,3]).torso_joint   # returns 0
        #         #           JointPositions([0,1,2,3]).neck_swivel   # returns 1
        #         #           JointPositions([0,1,2,3]).as_list       # returns [0,1,2,3]
        #         JointPositions(
        #             torso_joint=torso_joint, # NOTE: units = degrees
        #             neck_swivel=neck_swivel, # <- more negative means more to your left side (the survivor buddy's right side)
        #             head_tilt=head_tilt, # idk which way is which, but tilt is the "roll" in "yaw, pitch, roll"
        #             head_nod=-head_nod, # <- more negative = face the cieling
        #         )
        #     )
        #     # Robot.tell_camera_server({"torso": torso_joint,
        #     #                         "neck": neck_swivel,
        #     #                         "head_tilt": head_tilt,
        #     #                         "head_nod": head_nod})
        
    
    def when_video_chunk_received(chunk):
        numpy_image_array = cv2.imdecode(numpy.frombuffer(chunk.data, np.uint8), cv2.IMREAD_COLOR)
        
        clapBehavior.visualData = numpy_image_array
        clapBehavior.performBehavior()
            
            
            
    
    def setup_all_the_boring_boilerplate_stuff():
        # NOTE: read this function if you want to know how ROS actually works
        rospy.init_node('main_survivor_buddy_node', anonymous=True)
        
        Robot.joint_publisher = rospy.Publisher(
            "/sb_cmd_state",
            TwistStamped,
            queue_size=20
        )
        Robot.face_publisher = rospy.Publisher(
            "/camera_server/do_something",
            std_msgs.msg.String,
            queue_size=5,
        )
        if config.send_to_rviz:
            Robot.movement_publisher = rospy.Publisher(
                "/move_group/display_planned_path",
                DisplayTrajectory,
                queue_size=20
            )
        rospy.loginfo("Node started.")

        Robot.twist_obj = TwistStamped()
        Robot.previous_joint_positions = JointPositions(
            torso_joint=0, # degrees not radians
            neck_swivel=0, # degrees not radians
            head_tilt=0, # degrees not radians
            head_nod=0, # degrees not radians
        )
        if config.send_to_rviz:
            Robot.robot = moveit_commander.RobotCommander()
            Robot.scene = moveit_commander.PlanningSceneInterface()

            Robot.group_name = "survivor_buddy_head"
            Robot.move_group = moveit_commander.MoveGroupCommander(Robot.group_name)
        
            Robot.previous_joint_positions = JointPositions(Robot.move_group.get_current_joint_values())
            Robot.display_trajectory = DisplayTrajectory()
        
        Robot.has_initialized = True
        
        # 
        # setup listeners
        # 
        if True:
            Robot.audio_subscriber = rospy.Subscriber(
                "/audio",
                Float32MultiArray,
                callback=Robot.when_audio_chunk_received,
                queue_size=1
            )
            Robot.camera_subscriber = rospy.Subscriber(
                "/camera/image/compressed",
                CompressedImage,
                callback=Robot.when_video_chunk_received,
                queue_size=1
            )
    
    def tell_camera_server(data):
        # NOTE: you probably dont want to edit me
        import json
        Robot.face_publisher.publish(
            std_msgs.msg.String(
                json.dumps(data)
            )
        )

    def move_towards_positions(joint_goals, *, wait=False):
        # NOTE: you probably dont want to edit me
        if not isinstance(joint_goals, JointPositions):
            raise Exception(f'''
                When calling Robot.move_towards_positions()
                    make sure the first argument is a JointPositions object
                    Ex:
                        Robot.move_towards_positions(
                            JointPositions(
                                torso_joint=5, # NOTE: units = degrees
                                neck_swivel=5, # <- more negative means more to your left side (the survivor buddy's right side)
                                head_tilt=5, # idk which way is which, but tilt is the "roll" in "yaw, pitch, roll"
                                head_nod=5, # <- more negative = face the cieling
                            )
                        )
            ''')
        
        Robot.previous_joint_positions = joint_goals = JointPositions(joint_goals.as_list)
        
        if not config.send_to_rviz:
            send_to_survivor_bud(joint_goals, speed=30)
            if 0:
                Robot.twist_obj.twist.linear.x  = -joint_goals.torso_joint
                Robot.twist_obj.twist.linear.y  = -joint_goals.neck_swivel
                Robot.twist_obj.twist.linear.z  =  joint_goals.head_tilt
                Robot.twist_obj.twist.angular.x = -joint_goals.head_nod
                Robot.joint_publisher.publish(Robot.twist_obj)
                Robot.previous_joint_positions = joint_goals
        else:
            joint_current = Robot.move_group.get_current_joint_values()
            
            joint_current[0] = joint_goals.torso_joint
            joint_current[1] = joint_goals.neck_swivel
            joint_current[2] = joint_goals.head_tilt
            joint_current[3] = joint_goals.head_nod
            
            # this gets rid of standing-still "jitter"
            if all(prev == current for prev, current in zip(Robot.previous_joint_positions.as_list, joint_current)):
                return
            else:
                Robot.previous_joint_positions = JointPositions(joint_current)
            
            Robot.move_group.go(tuple(math.radians(each) for each in joint_current), wait=wait)
            plan = Robot.move_group.plan()
            Robot.move_group.stop()
            
            if config.send_to_rviz: # its a lot faster/smoother movement when not enabling simulation
                Robot.display_trajectory = DisplayTrajectory()
                Robot.display_trajectory.trajectory_start = Robot.robot.get_current_state()
                Robot.movement_publisher.publish(Robot.display_trajectory)

            # execute plan
            Robot.move_group.execute(plan[1], wait=wait)


# 
# commandline arguments
# 


if True:
    # convert stuff like "--send_to_rviz False" into { "send_to_rviz": False }
    arg_list, cli_args_as_dict = convert_args(sys.argv)
    # make the cli args override the config
    config.update(cli_args_as_dict)

# Starting point of everything
Robot.setup_all_the_boring_boilerplate_stuff()
moveit_commander.roscpp_initialize([])
rospy.spin()
