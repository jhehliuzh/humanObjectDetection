import os
import sys


sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from _util.make_folder_dataset import MakeFolderDataset
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from _util.util import choose_dataset, user_input_choose_from_list

majority_voting_len = 15
contact_n = 2

# create classification window testing dataframe
cwt_path = Path(os.environ.get("DATASET_REPO_ROOT_PATH")) / \
    "testData" / "classification_window_testing"
cwt_instance_path = user_input_choose_from_list([p.name for p in cwt_path.iterdir(
) if p.is_dir()], "classification window testing instances")
cwt_instance_path = cwt_path / cwt_instance_path

cwt_path = cwt_instance_path / "classification_windows.csv"
classification_windows = pd.read_csv(str(cwt_path.absolute()), header=None)

columns = ["time_sec", "time_nsec", "majority_vote_counter"]
for i in range(7):
    for j in range(40):
        columns.append(f"tau_J{i}_{j}")
for i in range(7):
    for j in range(40):
        columns.append(f"e_q{i}_{j}")
for i in range(7):
    for j in range(40):
        columns.append(f"e_dq{i}_{j}")
classification_windows.columns = columns

# calculate time and shift it to start at 0
classification_windows["time"] = classification_windows["time_sec"] + \
    classification_windows["time_nsec"]
classification_windows['time'] = classification_windows['time'] - \
    classification_windows['time'][0]

# shift majority voting counter to start at 0
min_value = classification_windows["majority_vote_counter"].min()
classification_windows["majority_vote_counter"] -= min_value

classification_windows = classification_windows.groupby(
    "majority_vote_counter").filter(lambda x: len(x) >= majority_voting_len)
majority_votes = [int(
    counter) for counter in classification_windows["majority_vote_counter"].unique()]
majority_vote_n = user_input_choose_from_list(majority_votes, "majority votes")

classification_windows = classification_windows[
    classification_windows["majority_vote_counter"] == majority_vote_n]

filtered_columns = [
    col for col in classification_windows.columns if col.startswith('tau_J0')]
classification_windows = classification_windows[filtered_columns]

for i in range(len(classification_windows)):
    plt.figure(i)
    plt.plot(classification_windows.iloc[i, :], label="online",
             linestyle='-', color='blue', linewidth=2)
    plt.legend()

# print(reshaped_array.shape)
# print(reshaped_array)

# processed data classification windows
prediction_step_size = 3
classification_model_input_size = 21
window_classification_length = 40

processed_x_file = Path(os.environ.get("DATASET_REPO_ROOT_PATH")) / \
    "processedData" / "TESTSET" / f"x_{cwt_instance_path.name}.npy"
X = np.load(str(processed_x_file.absolute()))
y = np.load(str((processed_x_file.parent /
            processed_x_file.name.replace("x_", "y_")).absolute()))

# filter out data points based on prediction step size: only keep data for every n-th timestep, so offline test simulates predicting at every n-th timestep
timestep_mask = (y[:, 2].astype(int) +
                 window_classification_length) % prediction_step_size == 0
y = y[timestep_mask]
X = X[timestep_mask]

# filter X features to fit model
target_torque = ['tau_J0', 'tau_J1', 'tau_J2',
                 'tau_J3', 'tau_J4', 'tau_J5', 'tau_J6']
target_position_err = ['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']
target_velocity_err = ['de0', 'de1', 'de2', 'de3', 'de4', 'de5', 'de6']
model_features = target_torque + target_position_err + target_velocity_err

dataset_targets = np.load(str((Path(os.environ.get(
    "DATASET_REPO_ROOT_PATH")) / "processedData" / "targets.npy").absolute()))

feature_indices = np.where(np.isin(dataset_targets, model_features))[0]
X = X[:, :, feature_indices]
X_test_tensor = torch.tensor(X, dtype=torch.float32)  # RNN input dimensions


encoder = LabelEncoder()
y[:, 0] = encoder.fit_transform(y[:, 0])

# group X data by contact to simulate individual contact predictions for majority voting
unique_inst_contact_ids = np.unique(y[:, 1])
X_grouped_by_contact = [X_test_tensor[y[:, 1] == i]
                        for i in unique_inst_contact_ids]
y_by_contact = np.array([int(y[y[:, 1] == i][0][0])
                        for i in unique_inst_contact_ids])

X_contact_tensor = X_grouped_by_contact[majority_vote_n]
# start_idx = (majority_voting_len * majority_vote_n)
for i, X_window_at_timestep in enumerate(X_contact_tensor[0:majority_voting_len].unsqueeze(1)):
    plt.figure(i)
    plt.plot(X_window_at_timestep[:, :, 0].squeeze(
    ), label="offline", linestyle="--", color='orange', linewidth=2)
    plt.legend()
    # plt.title(f"offline - window {i}")

plt.show()
