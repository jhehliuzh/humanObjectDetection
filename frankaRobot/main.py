#!/usr/bin/env python3
"""
By Maryam Rezayati

# How to run?
1. unlock robot
	-turn on the robot (wait until it has a solid yellow)
	-connect to the robot desk with the ID (172.16.0.2 or 192.168.15.33)
	-unlock the robot
	-the robot light should be blue
	-unlock the robot and activate FCI

2. run frankapy
	-open an terminal
		conda activate frankapyenv
		bash /home/mindlab/franka/frankapy/bash_scripts/start_control_pc.sh -i localhost

3. run sensor node
	-open another temrinal
		source /opt/ros/noetic/setup.bash
		/home/mindlab/miniconda3/envs/frankapyenv/bin/python /home/mindlab/humanObjectDetection/dataLabeling/sensorNode.py

4. run robot node
		conda activate frankapyenv
		source /opt/ros/noetic/setup.bash
		source /home/mindlab/franka/franka-interface/catkin_ws/devel/setup.bash --extend
		source /home/mindlab/franka/frankapy/catkin_ws/devel/setup.bash --extend
	
		/home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/humanObjectDetection/frankaRobot/main.py
	-open another terminal 

5. run save data node
	-open another terminal
		source /opt/ros/noetic/setup.bash
		source /home/mindlab/franka/franka-interface/catkin_ws/devel/setup.bash --extend
		source /home/mindlab/franka/frankapy/catkin_ws/devel/setup.bash --extend
	
		/home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/humanObjectDetection/frankaRobot/saveDataNode.py

# to chage publish rate of frankastate go to : 
sudo nano /home/mindlab/franka/franka-interface/catkin_ws/src/franka_ros_interface/launch/franka_ros_interface.launch
"""

## import required libraries 
import os
import numpy as np
import pandas as pd
import time

import torch
from torchvision import transforms

import rospy
from std_msgs.msg import Float64
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from frankapy import FrankaArm
from franka_interface_msgs.msg import RobotState
from threading import Thread
from threading import Event
from importModel import import_lstm_models

# Set the main path
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'

# Define parameters for the LSTM models
num_features_lstm = 4
#contact_detection_path= main_path +'AIModels/trainedModels/contactDetection/trainedModel_06_30_2023_10_16_53.pth'
contact_detection_path= main_path +'AIModels/trainedModels/contactDetection/trainedModel_01_24_2024_11_18_01.pth'

#collision_detection_path = main_path + 'AIModels/trainedModels/collisionDetection/trainedModel_06_30_2023_09_07_24.pth'
collision_detection_path = main_path + 'AIModels/trainedModels/collisionDetection/trainedModel_01_24_2024_11_12_30.pth'

#localization_path = main_path + 'AIModels/trainedModels/localization/trainedModel_06_30_2023_09_08_08.pth'
localization_path = main_path + 'AIModels/trainedModels/localization/trainedModel_01_24_2024_11_15_06.pth'


window_length = 28
features_num = 28
dof = 7

# Define paths for joint motion data
joints_data_path = main_path + 'frankaRobot/robotMotionPoints/robotMotionJointData_b11.csv'
#joints_data_path = main_path + 'frankaRobot/robotMotionPoints/robotMotionJointData_c4.csv'

# load model
model_contact, labels_map_contact = import_lstm_models(PATH=contact_detection_path, num_features_lstm=num_features_lstm)
model_collision, labels_map_collision = import_lstm_models(PATH=collision_detection_path, num_features_lstm=num_features_lstm)
model_localization, labels_map_localization = import_lstm_models(PATH=localization_path, num_features_lstm=num_features_lstm)

# Set device for PyTorch models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == "cuda":
	torch.cuda.get_device_name()

# Move PyTorch models to the selected device
model_contact = model_contact.to(device)
model_collision = model_collision.to(device)
model_localization = model_localization.to(device)

# Define transformation for input data
transform = transforms.Compose([transforms.ToTensor()])
window = np.zeros([window_length, features_num])

# Create message for publishing model output (will be used in saceDataNode.py)
model_msg = Floats()

# Callback function for contact detection
def contact_detection(data):
	global window, publish_output, big_time_digits
	start_time = rospy.get_time()
	e_q = np.array(data.q_d) - np.array(data.q)
	e_dq = np.array(data.dq_d ) - np.array(data.dq)
	tau_J = np.array(data.tau_J) # we also have tau_J_d
	tau_ext = np.array(data.tau_ext_hat_filtered)
	tau_ext = np.multiply(tau_ext, 0.5)

	new_row = np.hstack((tau_J, tau_ext, e_q, e_dq))
	new_row = new_row.reshape((1,features_num))

	window = np.append(window[1:,:],new_row, axis=0)

	#change the order for lstm	
	lstmDataWindow = []
	for j in range(dof):
		# tau(t), tau_ext(t), e(t), de(t)

		if num_features_lstm == 4:
			column_index = [j, j+dof, j+dof*2, j+dof*3 ]
		elif num_features_lstm ==2:
			column_index = [j+dof*2, j+dof*3 ]
		elif num_features_lstm ==3:
			column_index = [j+dof, j+dof*2, j+dof*3 ]

		join_data_matix = window[:, column_index]
		lstmDataWindow.append(join_data_matix.reshape((1, num_features_lstm*window_length)))

	lstmDataWindow = np.vstack(lstmDataWindow)

	with torch.no_grad():
		data_input = transform(lstmDataWindow).to(device).float()
		model_out = model_contact(data_input)
		model_out = model_out.detach()
		output = torch.argmax(model_out, dim=1)

	contact = output.cpu().numpy()[0]
	if contact == 1:
		#with torch.no_grad():
			#model_out = model_collision(data_input)
			#model_out = model_out.detach()
			#output = torch.argmax(model_out, dim=1)
			#collision = output.cpu().numpy()[0]
#
			#model_out = model_localization(data_input)
			#model_out = model_out.detach()
			#output = torch.argmax(model_out, dim=1)
			#localization = output.cpu().numpy()[0]
		detection_duration  = rospy.get_time()-start_time
		rospy.loginfo('detection duration: %f, There is a: %s on %s',detection_duration, 0, 0)
		#rospy.loginfo(np.array([detection_duration, contact, collision, localization, window]))
		#publish_output.publish([detection_duration, contact, collision, localization])
		collision = contact
		localization = contact
		

	else:
		detection_duration  = rospy.get_time()-start_time
		collision = contact
		localization = contact
		rospy.loginfo('detection duration: %f, there is no contact',detection_duration)
		#publish_output.publish([detection_duration, contact, contact, contact])
	start_time = np.array(start_time).tolist()
	time_sec = int(start_time)
	time_nsec = start_time-time_sec
	model_msg.data = np.append(np.array([time_sec-big_time_digits, time_nsec, detection_duration, contact, collision, localization], dtype=np.complex128), np.hstack(window))
	#model_pub.publish(model_msg)
	


def move_robot(fa:FrankaArm, event: Event):

	joints = pd.read_csv(joints_data_path)

	# preprocessing
	joints = joints.iloc[:, 1:8]
	joints.iloc[:,6] -= np.deg2rad(45) 
	print(joints.head(5), '\n\n')
	fa.goto_joints(np.array(joints.iloc[0]),ignore_virtual_walls=True)
	fa.goto_gripper(0.02)
	
	while True:	
		try:	
			for i in range(joints.shape[0]):
				fa.goto_joints(np.array(joints.iloc[i]),ignore_virtual_walls=True,duration=2)
				#time.sleep(0.01)

		except Exception as e:
			print(e)
			event.set()
			break
	
	print('fininshed .... !')



if __name__ == "__main__":
	global publish_output, big_time_digits
	event = Event()
	# create robot controller instance
	fa = FrankaArm()
	scale = 1000000
	big_time_digits = int(rospy.get_time()/scale)*scale
	# subscribe robot data topic for contact detection module
	rospy.Subscriber(name= "/robot_state_publisher_node_1/robot_state",data_class= RobotState, callback =contact_detection)#, callback_args=update_state)#,queue_size = 1)
	model_pub = rospy.Publisher("/model_output", numpy_msg(Floats), queue_size= 1)
	move_robot(fa, event)
	
