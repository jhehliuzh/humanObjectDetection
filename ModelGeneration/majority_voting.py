from abc import ABC, abstractmethod

import numpy as np
import torch
#import rospy


class MajorityVotingClassifier(ABC):

    def __init__(self, model, nof_individual_predictions, output_size):
        self.model = model
        self.nof_individual_predictions = nof_individual_predictions
        self.output_size = output_size
        self.probability_window = []

    def update_probabilities(self, new_probabilities):
        self.probability_window.append(new_probabilities)
        if len(self.probability_window) > self.nof_individual_predictions:
            self.probability_window.pop(0)

    def majority_voting(self):
        if not self.probability_window or len(self.probability_window) < self.nof_individual_predictions:
            return None
        return self.do_majority_voting()

    def predict(self, data):
        # don't append data and predict if window is  full
        if len(self.probability_window) >= self.nof_individual_predictions:
            return None

        with torch.no_grad():
            output = self.model(data)
            prob  = torch.nn.functional.sigmoid(output) if self.output_size == 1 else torch.nn.functional.softmax(output, dim=1)            
        self.update_probabilities(prob.detach().cpu())

        # don't predict if window is not full
        if len(self.probability_window) < self.nof_individual_predictions:
            return None
        return self.majority_voting()

    def reset(self):
        self.probability_window = []

    def get_has_predicted(self):
        return len(self.probability_window) >= self.nof_individual_predictions

    @abstractmethod
    def do_majority_voting(self):
        pass


class SoftVotingClassifier(MajorityVotingClassifier):

    def __init__(self, model, nof_individual_predictions, output_size):
        super().__init__(model, nof_individual_predictions, output_size)

    def do_majority_voting(self):
        #print(f"soft voting probability window: {self.probability_window}")
        if self.output_size == 1:
            avg_prob = torch.mean(torch.stack(self.probability_window)).item()
            final_prediction = avg_prob > 0.5
            #rospy.loginfo(f"prob_window: {self.probability_window}")
        else:
            average_probabilities = torch.mean(
                torch.stack(self.probability_window), dim=0)
            final_prediction = torch.argmax(average_probabilities).item()
        return final_prediction


class HardVotingClassifier(MajorityVotingClassifier):

    def __init__(self, model, nof_individual_predictions, output_size):
        super().__init__(model, nof_individual_predictions, output_size)

    def do_majority_voting(self):
        #print(f"hard voting probability window: {self.probability_window}")
        if self.output_size == 1:
            ones_count = (torch.tensor(self.probability_window) > 0.5).sum().item()
            return 1 if ones_count > (len(self.probability_window) / 2) else 0
        else:
            predictions = [torch.argmax(probabilities, dim=1).item()
                           for probabilities in self.probability_window]
            final_prediction = np.bincount(predictions).argmax()
            return final_prediction
