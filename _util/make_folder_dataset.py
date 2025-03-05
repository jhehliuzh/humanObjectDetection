from pathlib import Path
import pandas as pd
import numpy as np
import json


class MakeFolderDataset:
    def __init__(self, folder_path: Path) -> None:
        self.path = folder_path
        self.num_lines_per_message = 130
        self.df = pd.DataFrame()
        self.df_dataset = pd.DataFrame()
        self.true_label = pd.DataFrame()
        self.tau = ['tau_J0', 'tau_J1', 'tau_J2',
                    'tau_J3', 'tau_J4', 'tau_J5', 'tau_J6']
        self.tau_d = ['tau_J_d0', 'tau_J_d1', 'tau_J_d2',
                      'tau_J_d3', 'tau_J_d4', 'tau_J_d5', 'tau_J_d6']
        self.tau_ext = ['tau_ext0', 'tau_ext1', 'tau_ext2',
                        'tau_ext3', 'tau_ext4', 'tau_ext5', 'tau_ext6']

        self.q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']
        self.q_d = ['q_d0', 'q_d1', 'q_d2', 'q_d3', 'q_d4', 'q_d5', 'q_d6']

        self.dq = ['dq0', 'dq1', 'dq2', 'dq3', 'dq4', 'dq5', 'dq6']
        self.dq_d = ['dq_d0', 'dq_d1', 'dq_d2',
                     'dq_d3', 'dq_d4', 'dq_d5', 'dq_d6']

        self.e = ['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']
        self.de = ['de0', 'de1', 'de2', 'de3', 'de4', 'de5', 'de6']
        self.etau = ['etau_J0', 'etau_J1', 'etau_J2',
                     'etau_J3', 'etau_J4', 'etau_J5', 'etau_J6']
        
        self.init_time = 0
        self.name = folder_path.name
        self.model_results = pd.DataFrame()

        with open(str((self.path / 'meta.json').absolute())) as meta_json:
            meta_data = json.load(meta_json)
            self.contact_type = meta_data['contact_type']
            self.start_from_time = meta_data['start_from_time'] if 'start_from_time' in meta_data else -1
            self.reference_duration_multiplier_lower = meta_data[
                'reference_duration_multiplier_lower'] if 'reference_duration_multiplier_lower' in meta_data else None
            self.reference_duration_multiplier_upper = meta_data[
                'reference_duration_multiplier_upper'] if 'reference_duration_multiplier_upper' in meta_data else None
            self.motion = meta_data["motion_filename"]
            
            self.dynamic = 1 if 'dynamic' in meta_data and meta_data["dynamic"] else 0

    def _extract_array(self, data_dict: dict, data_frame: str, header: list,  n: int):
        _, y = data_frame[n].split(':')
        y = y.replace('[', '')
        y = y.replace(']', '')
        y = y.replace('\n', '')
        y = y.split(',')
        for i, h in enumerate(header):
            data_dict[h].append(float(y[i]))

    def extract_robot_data(self):
        # it extracts robot data from all_data.txt
        all_data_path_str = str((self.path / 'all_data.txt').absolute())
        f = open(all_data_path_str, 'r')
        lines = f.readlines()

        keywords = ['time'] + self.tau + self.tau_d + \
            self.tau_ext + self.q + self.q_d + self.dq + self.dq_d
        data_dict = dict.fromkeys(keywords)
        for i in keywords:
            data_dict[i] = [0]

        for i in range(int(len(lines)/self.num_lines_per_message)):
            data_frame = lines[i *
                               self.num_lines_per_message:(i+1)*self.num_lines_per_message]

            # extract exact time from file sec + nsec
            _, y = data_frame[3].split(':')
            time_ = int(y)-int(int(y)/1000000)*1000000

            _, y = data_frame[4].split(':')
            time_ = time_+int(y)/np.power(10, 9)

            data_dict['time'].append(time_)

            self._extract_array(data_dict, data_frame, self.tau, 25)
            self._extract_array(data_dict, data_frame, self.tau_d, 26)
            self._extract_array(data_dict, data_frame, self.tau_ext, 37)

            self._extract_array(data_dict, data_frame, self.q, 28)

            self._extract_array(data_dict, data_frame, self.q_d, 29)
            self._extract_array(data_dict, data_frame, self.dq, 30)
            self._extract_array(data_dict, data_frame, self.dq_d, 31)

        self.df = pd.DataFrame.from_dict(data_dict)
        self.df = self.df.drop(index=0).reset_index(drop=True)
        for i in range(len(self.e)):
            self.df[self.e[i]] = self.df[self.q_d[i]]-self.df[self.q[i]]
            self.df[self.de[i]] = self.df[self.dq_d[i]]-self.df[self.dq[i]]
            self.df[self.etau[i]] = self.df[self.tau_d[i]]-self.df[self.tau[i]]

    def get_labels_all(self):
        true_label_path_str = str((self.path / 'true_label.csv').absolute())
        self.true_label = pd.read_csv(true_label_path_str)
        
        # ToDo @JHE: refactor this
        self.init_time = self.df['time'][0]

        self.true_label['time'] = self.true_label['time_sec'] + \
            self.true_label['time_nsec']-self.df['time'][0]
        self.df['time'] = self.df['time'] - self.df['time'][0]
        self.df['label'] = 0

        self.true_label['time_dev'] = self.true_label['time'].diff()

        # Find indices where 'time_dev' is greater than 0.05
        indices = self.true_label[self.true_label['time_dev'] > 0.05].index

        # Iterate through indices in reverse order to avoid index shifting issues
        for i in indices[::-1]:

            # Create a new row with the same values except for DATA0 which is set to 0
            new_row = self.true_label.iloc[i].to_frame().T
            new_row['DATA0'] = 0

            # Concatenate the DataFrame slices properly to avoid duplication
            self.true_label = pd.concat(
                [self.true_label.iloc[:i], new_row, self.true_label.iloc[i:]])

        # Reset index after concatenation
        self.true_label = self.true_label.reset_index(drop=True)

        # Sort the DataFrame by the 'time' column to maintain order
        self.true_label = self.true_label.sort_values(by='time')
