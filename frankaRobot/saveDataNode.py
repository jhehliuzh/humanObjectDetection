#!/usr/bin/env python3
"""
By Maryam Rezayat

This code continuously records data from the Franka Panda robot until manually stopped. It prompts the user to enter a tag name for labeling the data.

Files Created:
1. all_data.txt: Contains all data published by the Franka Panda robot.
2. true_label.csv: Contains data of true labels acquired through CANBUS communication.
3. model_result.csv: Presents the model output and data window.
4. meta.json: meta data about the run / saved data

How to Run:
1. Unlock the Robot:
	-turn on the robot (wait until it has a solid yellow)
	-connect to the robot desk with the ID (172.16.0.2 or 192.168.15.33)
	-unlock the robot
	-the robot light should be blue
	-unlock the robot and activate FCI

2. Connecting to the Robot (Running Frankapy):
	-open an terminal
		conda activate frankapyenv
		bash /home/mindlab/franka/frankapy/bash_scripts/start_control_pc.sh -i localhost

3. Run the Program:
    - Open another terminal:
        conda activate frankapyenv
        source /opt/ros/noetic/setup.bash
        source /home/mindlab/franka/franka-interface/catkin_ws/devel/setup.bash --extend
        source /home/mindlab/franka/frankapy/catkin_ws/devel/setup.bash --extend
        /home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/humanObjectDetection/frankaRobot/saveDataNode.py
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import csv
import json
from datetime import datetime
from pathlib import Path

# Import required libraries
import numpy as np
import rospy
from franka_interface_msgs.msg import RobotState
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Float64

from _util.util import choose_robot_motion, user_input_choose_from_list

# Set base path for saving data
subfolder = user_input_choose_from_list(
    choices=["rawData", "testData"], caption="subfolders")
ROOT_PATH = Path(f"/home/mindlab/humanObjectDetectionDataset/{subfolder}")

# Prompt the user to enter a tag name, the current robot motion and the contact type
FOLDER_TAG = input('Enter tag name: ')

used_robot_motion = choose_robot_motion()
contact_type = user_input_choose_from_list(
    choices=["soft", "hard", "pvc_tube"], caption="contact types")
dynamic = input("Is the contact dynamic? (y/n): ").lower() == 'y'


class LogData:
    def __init__(self) -> None:
        # Initialize ROS node
        rospy.init_node('log_data')
        print('ROS node is initiated!')

        # Create a folder for saving data
        self.PATH = ROOT_PATH / FOLDER_TAG
        os.mkdir(str(self.PATH.absolute()))

        # Create empty files for saving data
        all_data_path = str((self.PATH / 'all_data.txt').absolute())
        true_label_path = str((self.PATH / 'true_label.csv').absolute())
        model_results_path = str((self.PATH / 'model_result.csv').absolute())
        classification_windows_path = str(
            (self.PATH / 'classification_windows.csv').absolute())
        meta_path = str((self.PATH / 'meta.json').absolute())

        self.all_data_file = open(all_data_path, 'w')

        self.true_label_file = csv.writer(open(true_label_path, 'w'))
        self.true_label_file.writerow(
            ('time_sec', 'time_nsec', 'timestamp', 'DATA0'))

        self.model_results_file = csv.writer(open(model_results_path, 'w'))
        self.model_results_file.writerow(
            ('Time_sec', 'Time_nsec', 'prediction_duration', 'contact', 'contact_class_prediction', 'majority_voting_nof_individual_predictions'))

        self.classification_windows_file = csv.writer(
            open(classification_windows_path, 'w'))

        meta_data = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "contact_type": contact_type,
            "dynamic": dynamic,
            "start_from_time": -1,
            "motion_filename": used_robot_motion.name,
            "reference_duration_multiplier_lower": None,
            "reference_duration_multiplier_upper": None,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f, indent=4)

        print('*** Initialized files in ', str(self.PATH.absolute()), ' ***')

        # Subscribe to relevant ROS topics
        rospy.Subscriber(
            name="/model_output", data_class=numpy_msg(Floats), callback=self.save_model_output)
        rospy.Subscriber(
            name="/classification_window", data_class=numpy_msg(Floats), callback=self.save_classification_window)
        rospy.Subscriber(name="/robot_state_publisher_node_1/robot_state",
                         data_class=RobotState, callback=self.save_robot_state)
        rospy.Subscriber(name="/contactTimeIndex",
                         data_class=numpy_msg(Floats), callback=self.save_contact_index)

    def save_contact_index(self, data):
        # Save contact index data to true_label.csv
        data_row = np.array(data.data)
        self.true_label_file.writerow(data_row)

    def save_model_output(self, data):
        # Save model output data to model_result.csv
        data_row = np.array(data.data)
        self.model_results_file.writerow(data_row)

    def save_classification_window(self, data):
        data_row = np.array(data.data)
        self.classification_windows_file.writerow(data_row)

    def save_robot_state(self, data):
        # Save robot state data to all_data.txt
        self.all_data_file.write(str(data))


if __name__ == "__main__":
    # Create an instance of the LogData class
    log_data_instance = LogData()

    # Keep the program running to listen for ROS messages
    rospy.spin()
