import json
import os
from pathlib import Path
from typing import Any, Callable, Optional, Type, Union

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch

from ModelGeneration.rnn_models import RNNModel
from ModelGeneration.transformer.model import ConvTran


def get_repo_root_path():
    return Path(__file__).parents[1]


def user_input_choose_from_list(choices: "list[Any]", caption: str, list_item_label_selector: Optional[Callable[[Any], str]] = None):
    lines = [
        f'{i} {list_item_label_selector(v) if list_item_label_selector is not None else v}' for i, v in enumerate(choices)]
    print('\n' + '\n'.join(lines))
    index = None
    while index not in np.arange(0, len(choices), 1):
        index = int(input(f"Choose from {caption}: "))
    return choices[index]


def choose_robot_motion():
    robot_motions_path = get_repo_root_path() / "frankaRobot" / "robotMotionPoints"
    robot_motions = [p for _, p in enumerate(
        robot_motions_path.iterdir()) if p.is_file and p.suffix == ".csv"]
    return user_input_choose_from_list(robot_motions, "Robot Motions", lambda m: m.name)


def choose_dataset(subdir=None):
    processed_data_path = Path(os.environ.get("DATASET_REPO_ROOT_PATH"))
    processed_data_path = processed_data_path / subdir if subdir is not None else processed_data_path
    processed_data_path = processed_data_path / "processedData"

    subfolders = [str(p.name) for _, p in enumerate(processed_data_path.iterdir()) if p.is_dir()]
    subfolder = user_input_choose_from_list(subfolders, "subfolders")
    subfolder_path = processed_data_path / subfolder

    datasets = [str(p.name) for _, p in enumerate(subfolder_path.iterdir()) if p.is_file and p.name.startswith("x_") and p.suffix == ".npy"]
    dataset = user_input_choose_from_list(datasets, "datasets")
    dataset_path = subfolder_path / dataset

    return subfolder_path, dataset_path


def choose_normalization_mode():
    normalization_modes = [
        {"key": "", "caption": "No Normalization"},
        {"key": "S", "caption": "Standardization (mean/variance)"},
        {"key": "N", "caption": "Normalization (min/max)"}]
    return user_input_choose_from_list(normalization_modes, "Normalization Mode", lambda m: m["caption"])["key"]


def normalize_dataset(normalization_mode, data_train, data_test):
    scaler = None
    norm_mins, norm_maxes, norm_means, norm_vars = None, None, None, None

    def scale(scaler_type: type[Union[StandardScaler, MinMaxScaler]], data_train, data_test):
        scaler = scaler_type()
        N_train, S, D = data_train.shape
        N_test = data_test.shape[0]
        data_train = scaler.fit_transform(
            data_train.reshape(N_train * S, D)).reshape(N_train, S, D)
        data_test = scaler.transform(data_test.reshape(
            N_test * S, D)).reshape(N_test, S, D)
        return scaler, data_train, data_test

    if normalization_mode == "S":
        scaler, data_train, data_test = scale(StandardScaler, data_train, data_test)
        norm_means, norm_vars = scaler.mean_.tolist(), scaler.var_.tolist()
    elif normalization_mode == "N":
        scaler, data_train, data_test = scale(MinMaxScaler, data_train, data_test)
        norm_mins, norm_maxes = scaler.data_min_.tolist(), scaler.data_max_.tolist()

    return data_train, data_test, norm_mins, norm_maxes, norm_means, norm_vars, normalization_mode in ["S", "N"]


def choose_model_type():
    return user_input_choose_from_list(["RNN", "Transformer"], "Model Types")


def choose_trained_rnn_model(model_class: Type[RNNModel], trained_models_path=None):
    trained_models_path = get_repo_root_path() / "ModelGeneration" / "TrainedModels" if trained_models_path is None else trained_models_path
    with open(str((trained_models_path / "RnnModelsParameters.json").absolute()), 'r') as f:
        model_params = json.load(f)
    model_params = [
        m for m in model_params if m["model_name"].startswith(model_class.__name__ + "_")]
    model_params = sorted(
        model_params, key=lambda d: d['model_name'])
    return user_input_choose_from_list(model_params, "Trained model files", lambda v: v["model_name"])


def choose_trained_transformer_model():
    transformer_results_path = get_repo_root_path() / \
        "ModelGeneration" / "transformer" / "Results"
    trained_model_paths = []
    for sp in transformer_results_path.iterdir():
        if sp.is_dir() and sp.name != "old_models":
            trained_model_paths += [p for _,
                                    p in enumerate(sp.iterdir()) if p.is_dir()]
    return user_input_choose_from_list(trained_model_paths, "Trained Transformer models", lambda p: f"{p.parent.name}/{p.name}")


def load_rnn_classification_model(model_class: "type[RNNModel]", params, input_size, output_size, classification_dir=None):
    model_name = params["model_name"]
    classification_dir = get_repo_root_path() / "ModelGeneration" / "TrainedModels" if classification_dir is None else classification_dir
    classification_path = classification_dir /  f"{model_name}.pth"
    model_classification = model_class(
        input_size=input_size,
        hidden_size=params["hyperparameters"]["hidden_size"],
        num_layers=params["hyperparameters"]["num_layers"],
        output_size=output_size,
        dropout_rate=params["hyperparameters"]["dropout_rate"] if "dropout_rate" in params["hyperparameters"] else None)
    model_classification.load_state_dict(torch.load(
        str(classification_path.absolute()), map_location='cpu'))
    return model_classification


def load_transformer_classification_model(model_path: Path, input_size, output_size, window_length):
    checkpoint_path = model_path / "checkpoints" / "model_last.pth"
    config_path = model_path / "configuration.json"
    with open(str((config_path).absolute()), 'r') as f:
        config = json.load(f)
    config['Data_shape'] = [
        1, input_size, window_length]
    model_classification = ConvTran(
        config=config, num_classes=output_size)
    saved_params = torch.load(
        str(checkpoint_path.absolute()), map_location='cpu')
    model_classification.load_state_dict(saved_params["state_dict"])
    return model_classification, config


def normalize_window(window, params):
    if "normalization_max" in params and "normalization_min" in params and len(params["normalization_max"]) == window.shape[1] and len(params["normalization_min"]) == window.shape[1]:
        for i, max in enumerate(params["normalization_max"]):
            min = params["normalization_min"][i]
            window[:, i] = (window[:, i] - min) / (max - min)
    elif "normalization_mean" in params and "normalization_std" in params and len(params["normalization_mean"]) == window.shape[1] and len(params["normalization_std"]) == window.shape[1]:
        for i, mean in enumerate(params["normalization_mean"]):
            std = params["normalization_std"][i]
            window[:, i] = (window[:, i] - mean) / std
    return window
