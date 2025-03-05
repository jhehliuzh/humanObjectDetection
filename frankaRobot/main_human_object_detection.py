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
		/home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/humanObjectDetection/frankaRobot/main_human_object_detection.py


5. run save data node
	-open another terminal
		source /opt/ros/noetic/setup.bash
		source /home/mindlab/franka/franka-interface/catkin_ws/devel/setup.bash --extend
		source /home/mindlab/franka/frankapy/catkin_ws/devel/setup.bash --extend
		/home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/humanObjectDetection/frankaRobot/saveDataNode.py

# to chage publish rate of frankastate go to :
sudo nano /home/mindlab/franka/franka-interface/catkin_ws/src/franka_ros_interface/launch/franka_ros_interface.launch
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import signal
import threading
from threading import Event
from typing import Type

import numpy as np
import pandas as pd
import rospy
import torch
from franka_interface_msgs.msg import RobotState
from frankapy import FrankaArm
from importModel import import_lstm_models
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from torchvision import transforms

from _util.util import (choose_model_type, choose_robot_motion,
                        choose_trained_rnn_model,
                        choose_trained_transformer_model, get_repo_root_path,
                        load_rnn_classification_model,
                        load_transformer_classification_model,
                        normalize_window, user_input_choose_from_list)
from ModelGeneration.majority_voting import (HardVotingClassifier,
                                             SoftVotingClassifier)
from ModelGeneration.model_generation import choose_rnn_model_class

GREEN = '\033[92m'
RESET = '\033[0m'


class HumanObjectDetectionNode:
    def __init__(self):
        rospy.init_node('human_object_detection_node', disable_signals=True)

        self.shutdown_requested = Event()

        self.classification_model_input_size = 21
        self.window_classification_length = 40
        self.labels_classification = {0: "hard", 1: "pvc_tube", 2: "soft"}

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model_type = choose_model_type()
        self.model_classification, self.rnn_model_params, self.transformer_config = self.load_classification_model()
        self.model_classification = self.model_classification.to(self.device)
        self.model_classification.eval()

        majority_voting_classifier_class = user_input_choose_from_list(
            [HardVotingClassifier, SoftVotingClassifier], "Majority voting classifier", lambda v: v.__name__)
        nof_individual_predictions = int(user_input_choose_from_list(
            [8, 10, 12, 15], "Majority voting: umber of predictions"))
        self.majority_voting_classifier = majority_voting_classifier_class(
            model=self.model_classification, nof_individual_predictions=nof_individual_predictions, output_size=3)

        self.robot_motion_path = choose_robot_motion()

        self.classification_window = np.zeros(
            [self.window_classification_length, self.classification_model_input_size])
        self.model_output_msg = Floats()
        self.classification_window_msg = Floats()

        self.classification_counter = 0
        self.majority_vote_counter = 0
        self.has_contact = False

        self.fa = FrankaArm(init_node=False)
        self.scale = 1000000
        self.big_time_digits = int(rospy.get_time() / self.scale) * self.scale

        self.contact_timer = None

        self.robot_state_thread = threading.Thread(target=self.robot_state_listener)
        self.contact_thread = threading.Thread(target=self.contact_listener)       
        self.robot_state_thread.daemon = True
        self.contact_thread.daemon = True
        self.robot_state_thread.start()
        self.contact_thread.start()

        #rospy.Subscriber(name="/robot_state_publisher_node_1/robot_state",
        #                 data_class=RobotState, callback=self.contact_predictions)
        #rospy.Subscriber(name="/contactTimeIndex",
        #                 data_class=numpy_msg(Floats), callback=self.contact_cb)
        self.model_pub = rospy.Publisher(
            "/model_output", numpy_msg(Floats), queue_size=1)
        self.classification_window_pub = rospy.Publisher(
            "/classification_window", numpy_msg(Floats), queue_size=1)

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def robot_state_listener(self):
        rospy.Subscriber(name="/robot_state_publisher_node_1/robot_state",
                         data_class=RobotState, callback=self.contact_predictions)
        rospy.spin() 

    def contact_listener(self):
        rospy.Subscriber(name="/contactTimeIndex",
                         data_class=numpy_msg(Floats), callback=self.contact_cb)
        rospy.spin()

    def signal_handler(self, signum, frame):
        rospy.loginfo("Shutdown signal received.")
        self.shutdown_requested.set()
        rospy.signal_shutdown("Shutdown signal received")

    def load_classification_model(self):
        if self.model_type == "RNN":
            rnn_model_class = choose_rnn_model_class()
            rnn_model_params = choose_trained_rnn_model(rnn_model_class)
            model_classification = load_rnn_classification_model(
                rnn_model_class, rnn_model_params, self.classification_model_input_size, len(self.labels_classification))
            return model_classification, rnn_model_params, None
        elif self.model_type == "Transformer":
            transformer_model_path = choose_trained_transformer_model()
            model_classification, transformer_config = load_transformer_classification_model(
                transformer_model_path, self.classification_model_input_size, len(self.labels_classification), self.window_classification_length)
            return model_classification, None, transformer_config

    def contact_cb(self, _):
        self.has_contact = True
        self.restart_contact_timer()

    def contact_timer_callback(self, event):
        rospy.loginfo(
            "Timer callback triggered, set has_contact = False and reset majority voting classifier")
        self.has_contact = False
        self.majority_voting_classifier.reset()
        self.contact_timer.shutdown()

    def restart_contact_timer(self):
        if self.contact_timer is not None:
            self.contact_timer.shutdown()
        self.contact_timer = rospy.Timer(
            rospy.Duration(0.015), self.contact_timer_callback)

    def contact_predictions(self, data):
        if self.shutdown_requested.is_set():
            return

        start_time = rospy.get_time()

        e_q = np.array(data.q_d) - np.array(data.q)
        e_dq = np.array(data.dq_d) - np.array(data.dq)
        tau_J = np.array(data.tau_J)

        # Data for classification (human/object detection)
        classification_row = np.concatenate([tau_J, e_q, e_dq])
        classification_row = classification_row.reshape(
            (1, self.classification_model_input_size))
        self.classification_window = np.append(
            self.classification_window[1:, :], classification_row, axis=0)

        if self.has_contact == False:
            self.classification_counter = 0
            return

        # make classification prediction every 3rd time (classification_counter = 0, 1, 2)
        # only make predictions if majority voting classifier has not predicted yet (for this contact)
        has_predicted_for_contact = self.majority_voting_classifier.get_has_predicted()
        if self.classification_counter == 2 and not has_predicted_for_contact:
            start_time = np.array(start_time).tolist()
            time_sec = int(start_time)
            time_nsec = start_time - time_sec
            time_sec = time_sec - self.big_time_digits

            if self.model_type == "RNN":
                classification_tensor = torch.tensor(
                    self.classification_window, dtype=torch.float32).unsqueeze(0).to(self.device)
            elif self.model_type == "Transformer":
                classification_tensor = torch.tensor(
                    np.swapaxes(self.classification_window, 0, 1), dtype=torch.float32).unsqueeze(0).to(self.device)

            # majority voting: only returns prediction != None if this is the specified n-th prediction per contact
            majority_voting_prediction = self.majority_voting_classifier.predict(
                classification_tensor)

            flattened_classification_window = classification_tensor.numpy().flatten()
            self.classification_window_msg.data = np.concatenate([np.array(
                [time_sec, time_nsec, self.majority_vote_counter]), flattened_classification_window])
            self.classification_window_pub.publish(
                self.classification_window_msg)

            # publish prediction if majority voting classifier has predicted
            if majority_voting_prediction is not None:
                prediction_duration = rospy.get_time() - start_time
                rospy.loginfo(
                    GREEN +
                    f'prediction duration: {prediction_duration}, classification prediction: {self.labels_classification[int(majority_voting_prediction)]}'
                    + RESET)

                self.model_output_msg.data = np.array(
                    [time_sec, time_nsec, prediction_duration, 1, majority_voting_prediction, self.majority_voting_classifier.nof_individual_predictions])
                self.model_pub.publish(self.model_output_msg)

                self.majority_vote_counter += 1

        self.classification_counter = (self.classification_counter + 1) % 3

    def move_robot(self):
        joints = pd.read_csv(str(self.robot_motion_path.absolute()))

        # preprocessing
        joints = joints.iloc[:, 1:8]
        joints.iloc[:, 6] -= np.deg2rad(45)
        self.fa.goto_joints(
            np.array(joints.iloc[0]), ignore_virtual_walls=True)
        self.fa.goto_gripper(0.02)

        while not self.shutdown_requested.is_set():
            try:
                for i in range(joints.shape[0]):
                    if self.shutdown_requested.is_set():
                        break
                    self.fa.goto_joints(
                        np.array(joints.iloc[i]), ignore_virtual_walls=True, duration=1.5)
            except Exception as e:
                rospy.logerr(e)
                break

    def run(self):
        try:
            rospy.loginfo("Human Object Detection Node started.")
            self.move_robot()
            rospy.spin()
        except rospy.ROSInterruptException:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        rospy.loginfo("Shutting down Human Object Detection Node...")
        self.shutdown_requested.set()
        if self.contact_timer is not None:
            self.contact_timer.shutdown()
        rospy.loginfo("Shutdown complete.")


if __name__ == "__main__":
    node = HumanObjectDetectionNode()
    node.run()
