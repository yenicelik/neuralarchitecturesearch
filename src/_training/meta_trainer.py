"""
    This file includes the training, where we iteratively train
    1. the child model
    2. the controller (based on the loss of the child model)
"""
import torch

from src.child.networks.rnn.dag_rnn import dlxDAGRNNModule
from src.child.training.dag_train_wrapper import DAGTrainWrapper

class MetaTrainer:

    def __init__(self):
        pass

        # self.child = dlxDAGRNNModule()

    def train_child(self, dag):
        # This should be replaced by batch getters
        # Spawn child trainer and model
        model = dlxDAGRNNModule(dag)
        self.child_trainer = DAGTrainWrapper(model)

        X = torch.randn((401, 4, 50))
        self.child_trainer.train(X[:400, :], X[1:, :])

    def get_child_validation_loss(self):
        pass

    def train(self):

        # Setting up the trainers
        for current_epoch in range(3):
            dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
            dag_list = [int(x) for x in dag_description.split()]
            self.train_child(dag_list)


if __name__ == "__main__":
    print("Starting to train the meta model")
    meta_trainer = MetaTrainer()
    meta_trainer.train()