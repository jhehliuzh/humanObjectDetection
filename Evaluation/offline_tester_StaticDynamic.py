import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from _util.util import (choose_dataset, choose_model_type,
                        choose_trained_rnn_model,
                        choose_trained_transformer_model, get_repo_root_path,
                        load_rnn_classification_model,
                        load_transformer_classification_model,
                        normalize_window)
from ModelGeneration.majority_voting import (HardVotingClassifier,
                                             SoftVotingClassifier)
from ModelGeneration.model_generation import choose_rnn_model_class

classification_model_input_size = 21
window_classification_length = 40
labels_classification = {0: "static", 1: "dynamic"}


load_dotenv(find_dotenv())

model_type = choose_model_type()
model_classification, rnn_model_params, transformer_config = None, None, None
if model_type == "RNN":
    rnn_model_class = choose_rnn_model_class()
    rnn_model_params = choose_trained_rnn_model(rnn_model_class, get_repo_root_path(
    ) / "ModelGeneration" / "TrainedModels_StaticDynamic")
    model_classification = load_rnn_classification_model(
        rnn_model_class, rnn_model_params, classification_model_input_size, 1, get_repo_root_path() / "ModelGeneration" / "TrainedModels_StaticDynamic")
elif model_type == "Transformer":
    transformer_model_path = choose_trained_transformer_model()
    model_classification, transformer_config = load_transformer_classification_model(
        transformer_model_path, classification_model_input_size, len(labels_classification), window_classification_length)


_, X_file = choose_dataset(subdir="static_dynamic")

prediction_step_size = input(
    "\nprediction step size (predict at n-th timestep): ")
prediction_step_size = int(prediction_step_size)

X = np.load(str(X_file.absolute()))
y = np.load(str((X_file.parent / X_file.name.replace("x_", "y_")).absolute()))

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
    "DATASET_REPO_ROOT_PATH")) / "static_dynamic" / "processedData" / "targets.npy").absolute()))

feature_indices = np.where(np.isin(dataset_targets, model_features))[0]
X = X[:, :, feature_indices]

#encoder = LabelEncoder()
#y[:, 0] = encoder.fit_transform(y[:, 0])

# normalize
for i, x_i in enumerate(X):
    X[i] = normalize_window(
        x_i, rnn_model_params if model_type == "RNN" else transformer_config)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_classification = model_classification.to(device)
model_classification.eval()

if model_type == "Transformer":
    X_test_tensor = torch.tensor(np.swapaxes(
        X, 1, 2), dtype=torch.float32).to(device)
else:
    X_test_tensor = torch.tensor(X, dtype=torch.float32).to(device)


# group X data by contact to simulate individual contact predictions for majority voting
unique_inst_contact_ids = np.unique(y[:, 1])
X_grouped_by_contact = [X_test_tensor[y[:, 1] == i]
                        for i in unique_inst_contact_ids]
y_by_contact = np.array([int(y[y[:, 1] == i][0][0])
                        for i in unique_inst_contact_ids])

# initialize majority voting classifiers
# only include classifiers with majority window length <= available time steps in data
#majority_voting_window_lengths = [8, 10, 12, 15]
majority_voting_window_lengths = [12, 15, 18, 20]

min_steps = min([X_contact_tensor.shape[0]
                for X_contact_tensor in X_grouped_by_contact])
majority_voting_window_lengths = [
    l for l in majority_voting_window_lengths if l <= min_steps]

soft_voting_classifiers = [SoftVotingClassifier(
    model_classification, l, 1) for l in majority_voting_window_lengths]
hard_voting_classifiers = [HardVotingClassifier(
    model_classification, l, 1) for l in majority_voting_window_lengths]

# Perform inference with majority voting classifiers
soft_voting_predictions = [[]
                           for _ in range(len(majority_voting_window_lengths))]
hard_voting_predictions = [[]
                           for _ in range(len(majority_voting_window_lengths))]

for i in range(len(majority_voting_window_lengths)):
    for X_contact_tensor in X_grouped_by_contact:
        contact_predictions_soft, contact_predictions_hard = [], []
        for X_window_at_timestep in X_contact_tensor.unsqueeze(1):
            contact_predictions_soft.append(
                soft_voting_classifiers[i].predict(X_window_at_timestep))
            contact_predictions_hard.append(
                hard_voting_classifiers[i].predict(X_window_at_timestep))

        soft_voting_predictions[i].append(contact_predictions_soft)
        hard_voting_predictions[i].append(contact_predictions_hard)

        soft_voting_classifiers[i].reset()
        hard_voting_classifiers[i].reset()

# assert that for each majority voting prediction per contact, only the prediction at the given soft voting window length index is not None
# assert that for each majority voting prediction per contact, the prediction at the given soft voting window length index is a proper class label
for i in range(len(majority_voting_window_lengths)):
    for j in range(len(soft_voting_predictions[i])):
        predictions_soft, predictions_hard = soft_voting_predictions[
            i][j], hard_voting_predictions[i][j]
        for p in [predictions_soft, predictions_hard]:
            assert all(value is None for k, value in enumerate(
                p) if k != majority_voting_window_lengths[i] - 1), f"majority voting with window length {majority_voting_window_lengths[i]} made a prediction at a timestep other than the window length index"
            assert p[majority_voting_window_lengths[i] - 1] in [
                0, 1, 2], f"majority voting with window length {majority_voting_window_lengths[i]} made a prediction that is not a proper class label"

        # replace array of majority voting predictions with the prediction at the given window length index -> the one actual prediction of the majority voting classifier
        soft_voting_predictions[i][j] = [
            p for p in predictions_soft if p is not None][0]
        hard_voting_predictions[i][j] = [
            p for p in predictions_hard if p is not None][0]

soft_voting_predictions = np.array(soft_voting_predictions)
hard_voting_predictions = np.array(hard_voting_predictions)

# Calculate accuracies for all majority voting classifiers
accuracies_soft = [{"caption": f"soft voting classifier that evaluates first {l} predictions (~{l*5*prediction_step_size}ms w/ step size {prediction_step_size}) per contact",
                    "accuracy": np.mean(soft_voting_predictions[i] == y_by_contact)} for i, l in enumerate(majority_voting_window_lengths)]
accuracies_hard = [{"caption": f"hard voting classifier that evaluates first {l} predictions (~{l*5*prediction_step_size}ms w/ step size {prediction_step_size}) per contact",
                    "accuracy": np.mean(hard_voting_predictions[i] == y_by_contact)} for i, l in enumerate(majority_voting_window_lengths)]

# Save accuracies_soft and accuracies_hard as text files
model_name = rnn_model_params['model_name'] if model_type == "RNN" else f"Transformer_{transformer_model_path.parent.name}_{transformer_model_path.name}"
np.savetxt(get_repo_root_path() / "Evaluation" / "OfflineTestResults_StaticDynamic" / f"soft_voting_accuracies_{model_name}.txt", accuracies_soft, fmt='%s')
np.savetxt(get_repo_root_path() / "Evaluation" / "OfflineTestResults_StaticDynamic" / f"hard_voting_accuracies_{model_name}.txt", accuracies_hard, fmt='%s')

print(
    f"\nWindow size: {window_classification_length}, Prediction step size: {prediction_step_size}")
print("\nSoft Voting:")
for accuracy in accuracies_soft:
    print(
        f'Test Accuracy for {accuracy["caption"]}: {accuracy["accuracy"]:.4f}')
print("\nHard Voting:")
for accuracy in accuracies_hard:
    print(
        f'Test Accuracy for {accuracy["caption"]}: {accuracy["accuracy"]:.4f}')


def generate_confusion_matrices(majority_voting_predictions, cm_list):
    max_val = 0
    for predictions in majority_voting_predictions:
        cm_list.append(confusion_matrix(y_by_contact, predictions))
        local_max = np.max(cm_list[-1])
        if local_max > max_val:
            max_val = local_max
    return max_val


# Generate confusion matrices and extract max value to use as max value for all confusion matrices
confusion_matrices_soft, confusion_matrices_hard = [], []
confusion_matrices_max_val_soft = generate_confusion_matrices(
    soft_voting_predictions, confusion_matrices_soft)
confusion_matrices_max_val_hard = generate_confusion_matrices(
    hard_voting_predictions, confusion_matrices_hard)

class_labels = ['static', 'dynamic']


def display_confusion_matrix(confusion_matrix, ax, confusion_matrices_max_val, title):
    try:
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix, display_labels=class_labels)
        disp.plot(include_values=True, cmap='Blues',
                  ax=ax, xticks_rotation='horizontal')
        im = ax.images[0]
        im.set_clim(0, confusion_matrices_max_val)
        ax.set_title(title)
    except Exception as e:
        print(f"unable to create confusion matrix: {e}")


fig_soft, axs_soft = plt.subplots(2, 2, figsize=(10, 8))
axs_soft = axs_soft.flatten()
fig_hard, axs_hard = plt.subplots(2, 2, figsize=(10, 8))
axs_hard = axs_hard.flatten()

# display confusion matrices using same max value for each to normalize color scale
for i, cm in enumerate(confusion_matrices_soft):
    display_confusion_matrix(cm, axs_soft[i], confusion_matrices_max_val_soft,
                             f'soft voting: first {majority_voting_window_lengths[i]} predictions per contact\nprediction step size: {prediction_step_size}')

for i, cm in enumerate(confusion_matrices_hard):
    display_confusion_matrix(cm, axs_hard[i], confusion_matrices_max_val_hard,
                             f'hard voting: first {majority_voting_window_lengths[i]} predictions per contact\nprediction step size: {prediction_step_size}')

fig_soft.tight_layout()
fig_hard.tight_layout()
plt.tight_layout()
fig_soft.savefig(get_repo_root_path() / "Evaluation" / "OfflineTestResults_StaticDynamic" / f"soft_voting_CM_{model_name}.png", bbox_inches='tight')
fig_hard.savefig(get_repo_root_path() / "Evaluation" / "OfflineTestResults_StaticDynamic" / f"hard_voting_CM_{model_name}.png", bbox_inches='tight')
#plt.show()
