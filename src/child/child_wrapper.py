"""
    This is one example training wrapper which uses the interface given by the Base class.
    We will use the RNN cell as this provides the right interface for the NAS we will later implement
"""
import sys
import numpy as np
import torch
from torch import nn

from src._training.debug_utils.rnn_debug import print_batches
from src.config import C_DEVICE, config
from src.utils.debug_utils.size_network import memory_usage_resource
from src.model_config import ARG
from src.utils.debug_utils.tensorboard_tools import tx_writer, tx_counter

# Format the system output writer
toolbar_width = 40
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width + 1))


def _debug_memory_print(identifier):
    """
        Some print statements to print out how much memory the program uses
    :return:
    """
    if config['debug_memory']:
        print("Memory usage Loss {}: ".format(identifier), memory_usage_resource())


class ChildWrapper:

    def update_lr(self, lr):
        """
            Update the learning rate of the optimizer with the new learning rate
        :param lr:
        :return:
        """
        # Takes the optimizer, and the initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            print(param_group['lr'])

    def __init__(self, model):
        self.model = model

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=ARG.shared_lr,
            weight_decay=ARG.shared_l2_reg
        )

    # def predict(self, X, n=1):
    # TODO: Something important to check is if the weights give good predictions!
    #     """
    #         Predicts the next n elements given past X elements.
    #
    #             X.size() <- (total_data_size, time_length, **data_size )
    #
    #     :param X:
    #     :param Y:
    #     :return:
    #     """
    #
    #     assert n==1, ("Cases where n>1 are not implemented yet!", n)
    #
    #     # Take the very last output from a "forward"
    #     Y_hat = self.model.forward(X)
    #     prediction = Y_hat[:, -1]
    #     print("Predictions have size: ", prediction.size())
    #
    #     prediction_index = torch.argmax(prediction, dim=-1, keepdim=True)
    #     print("Prediction index has size: ", prediction_index.size())
    #
    #     tmax, _ = torch.max(prediction, dim=-1, keepdim=True)
    #     print(tmax)
    #     # print("Prediction and tmax shape: ", prediction.size(), tmax.size())
    #     e_x = torch.sub(prediction, tmax)
    #     class_probabilities = e_x / torch.sum(e_x, dim=-1, keepdim=True)
    #
    #     # print("Summed probabilites are: (should be all 1)", torch.sum(class_probabilities, dim=-1))
    #     print(class_probabilities.size())
    #
    #     return prediction_index, class_probabilities

    def _data_pass(self, X, Y):
        """
            Makes forward passes (with or without backward step
        :return:
        """

        loss_arr = []
        for data_idx in range(0, X.size(0), ARG.batch_size):

            if data_idx + ARG.batch_size > X.size(0):
                break

            _debug_memory_print(1)

            # Take subset of data, and apply all operations based on that
            X_cur = X[data_idx:data_idx + ARG.batch_size].to(C_DEVICE).detach()
            Y_cur = Y[data_idx:data_idx + ARG.batch_size].to(C_DEVICE).detach()

    def get_loss(self, X, Y):
        """
            Calculates the perplexity for the given X and Y's.
        :param X:
        :param Y:
        :return:
        """

        print("We're retrieving the loss of the following tensors: ")
        print(X.size(), Y.size())

        self.model.set_train(is_train=False)

        # Take the criterion iteratively because the entire data will not immediately fit into memory
        loss_arr = []
        for data_idx in range(0, X.size(0), ARG.batch_size):

            if data_idx + ARG.batch_size > X.size(0):
                break

            _debug_memory_print(1)

            # Take subset of data, and apply all operations based on that
            X_cur = X[data_idx:data_idx + ARG.batch_size].to(C_DEVICE).detach()
            Y_cur = Y[data_idx:data_idx + ARG.batch_size].to(C_DEVICE).detach()

            _debug_memory_print(2)

            # print("Size of the batches are: ", X_cur.size(), Y_cur.size())
            # print(data_idx, " from ", X.size(0))

            Y_hat = self.model.forward(X_cur)

            _debug_memory_print(3)

            Y_hat = Y_hat.transpose(1,
                                    -1).contiguous()  # TODO: Fix this thing of transposing randomly! Define the input dimension and feed it in like that
            Y_hat = Y_hat.transpose(2, -1).contiguous()
            # Y_hat = torch.argmax(Y_hat, len(Y_hat.size())-1)

            _debug_memory_print(4)

            Y_cur = Y_cur.transpose(1,
                                    -1).contiguous()  # TODO: Fix this thing of transposing randomly! Define the input dimension and feed it in like that
            Y_cur = Y_cur.transpose(2, -1).contiguous()
            Y_cur = Y_cur.squeeze()

            # print("Before entering criterion!", Y_hat.size(), Y_cur.size())
            _debug_memory_print(5)

            current_loss = self.criterion(Y_hat, Y_cur)

            print("Current loss is: ", current_loss)

            loss_arr.append(current_loss.item())

            _debug_memory_print(6)

        total_loss = sum(loss_arr) / len(loss_arr)

        return np.exp(total_loss) / Y.size(0)  # Gotta normalize the loss

    def train(self, X, Y):
        """
            Trains the model on a certain dataset.
            The datasets have to be of the shape:
            X.size() <- (total_data_size, time_length, **data_size )
            --> Watch out! There should be a cutoff and padding amongst batches!
            --> The total data doesn't have to be the entire data, but can be just
                the data size can be chosen
        :param X: The data
        :param Y: The shape
        :param log_offset: Only for logging! The offset at which we initialize the new child model
        :return:
        """

        self.model.set_train(is_train=True)

        assert X.size() == Y.size(), ("Not same size! (X, Y) :: ", X.size(), Y.size())

        data_size = X.size(0)
        losses = torch.empty(data_size // ARG.batch_size)

        # Do exactly one epoch
        for train_idx in range(0, data_size, ARG.batch_size):

            if train_idx + ARG.batch_size > data_size:
                break

            # print("Getting the individual batch for training")
            _debug_memory_print("L1")

            X_cur = X[train_idx:train_idx + ARG.batch_size, :].to(C_DEVICE).detach()
            Y_cur = Y[train_idx:train_idx + ARG.batch_size, :].to(C_DEVICE).detach()

            if config['debug_printbatch']:
                print_batches(X_cur, Y_cur)

            _debug_memory_print("L2")
            Y_hat = self.model.forward(X_cur)
            _debug_memory_print("L3")

            # TODO: Fix this thing of transposing randomly! Define the input dimension and feed it in like that
            Y_hat = Y_hat.transpose(1, -1).contiguous()
            Y_hat = Y_hat.transpose(2, -1).contiguous()

            # TODO: Fix this thing of transposing randomly! Define the input dimension and feed it in like that
            Y_cur = Y_cur.transpose(1, -1).contiguous()
            Y_cur = Y_cur.transpose(2, -1).contiguous()
            Y_cur = Y_cur.squeeze()

            _debug_memory_print("L4")

            loss = self.criterion(Y_hat, Y_cur)

            _debug_memory_print("L5")
            self.model.zero_grad()
            loss.backward()

            torch.cuda.empty_cache()

            _debug_memory_print("L6")

            # Clip gradients here
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), ARG.shared_grad_clip)
            self.optimizer.step()
            _debug_memory_print("L7")

            losses[train_idx // ARG.batch_size] = loss.item() / ARG.batch_size

            tx_counter[0] += 1
            tx_writer.add_scalar('loss/train_loss', loss.item() / ARG.batch_size, tx_counter[0])

            _debug_memory_print("L8")

            # Write the trailing load-line
            sys.stdout.write("-")
            sys.stdout.flush()

        sys.stdout.write("\n")
        print()

        losses = losses / (data_size // ARG.batch_size)
        print(losses)


if __name__ == "__main__":
    print("Do a bunch of forward passes: ")

    X = torch.LongTensor(401, 4, 1).random_(0, 10000)
    print(X.size())

    DAG_DESCRIPTION = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
    DAG_LIST = [int(x) for x in DAG_DESCRIPTION.split()]
    print(DAG_LIST)
    import src.child.networks.rnn.dag_rnn as dag_rnn

    # model = dlxExampleRNNModule()
    MODEL = dag_rnn.dlxDAGRNNModule()
    MODEL.overwrite_dag(DAG_LIST)

    TRAINER = ChildWrapper(MODEL)
    # Example forward pass
    # X = torch.randint((401, 4, 50))
    # trainer.train(X[:400,:], X[1:,:])

    TRAINER.predict(X[:400, :])
