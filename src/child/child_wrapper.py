"""
    This is one example training wrapper which uses the interface given by the Base class.
    We will use the RNN cell as this provides the right interface for the NAS we will later implement
"""
import sys
import numpy as np
import torch
from torch import nn

from src._training.debug_utils.rnn_debug import _print_batches
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

    def _data_pass(self, X, Y, identifier_prefix="", apply_backward=False, verbose_loss=False):
        """
            Makes forward passes (with or without backward step) through a given dataset.
            Each step goes in batches (as defined by ARG.batch_size) .
        :return:
        """

        assert X.size() == Y.size(), ("Not the same size! (X, Y) :: ", X.size(), Y.size())

        data_size = X.size(0)

        loss_arr = torch.empty(data_size // ARG.batch_size)
        for data_idx in range(0, X.size(0), ARG.batch_size):

            if data_idx + ARG.batch_size > X.size(0):
                break

            _debug_memory_print(identifier_prefix + str(1))

            # Take subset of data, and apply all operations based on that
            X_cur = X[data_idx:data_idx + ARG.batch_size].to(C_DEVICE).detach()
            Y_cur = Y[data_idx:data_idx + ARG.batch_size].to(C_DEVICE).detach()

            _print_batches(X_cur, Y_cur)
            _debug_memory_print(identifier_prefix + str(2))

            Y_hat = self.model.forward(X_cur)
            _debug_memory_print(identifier_prefix + str(3))


            # TODO: This feels a bit like randomly permuting the stuff
            Y_hat = Y_hat.transpose(1, -1).contiguous()
            Y_hat = Y_hat.transpose(2, -1).contiguous()

            Y_cur = Y_cur.transpose(1, -1).contiguous()
            Y_cur = Y_cur.transpose(2, -1).contiguous()
            Y_cur = Y_cur.squeeze()
            _debug_memory_print(identifier_prefix + str(4))

            # TODO: normalize loss appropriately
            loss = self.criterion(Y_hat, Y_cur)
            _debug_memory_print(identifier_prefix + str(5))

            if verbose_loss:
                print("Current {} loss: ".format(identifier_prefix), loss)

            # Optional optimization of the parameters
            if apply_backward:

                # Zero out previous gradients
                self.model.zero_grad()
                loss.backward()

                torch.cuda.empty_cache()
                _debug_memory_print(identifier_prefix + str(6))

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    ARG.shared_grad_clip
                )

                # Step towards more optimal step
                self.optimizer.step()
                _debug_memory_print(identifier_prefix + str(7))

            loss_arr[data_idx // ARG.batch_size] = loss.item() / ARG.batch_size

            tx_counter[0] += 1
            # TODO: Add the actual perplexity loss here!
            tx_writer.add_scalar(
                'loss/{}'.format(identifier_prefix),
                loss.item() / ARG.batch_size,
                tx_counter[0]
            )
            _debug_memory_print(identifier_prefix + str(8))

            # Write the trailing load-line
            if not verbose_loss:
                sys.stdout.write("-")
                sys.stdout.flush()

        sys.stdout.write("\n")
        print()

        loss_arr = loss_arr / (data_size // ARG.batch_size)
        print(loss_arr)

        return loss_arr

    def get_loss(self, X, Y, id_prefix):
        """
            Calculates the perplexity for the given X and Y's.
        :param X:
        :param Y:
        :return:
        """

        print("We're retrieving the loss of the following tensors: ")
        print(X.size(), Y.size())

        self.model.set_train(is_train=False)

        ###### Make the actual forward pass
        loss_arr = self._data_pass(
            X=X,
            Y=Y,
            identifier_prefix=id_prefix,
            apply_backward=False,
            verbose_loss=True
        )

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

        loss_arr = self._data_pass(
            X=X,
            Y=Y,
            identifier_prefix="train",
            apply_backward=True,
            verbose_loss=False
        )

        # TODO: before this, the loss was different!
        total_loss = sum(loss_arr) / len(loss_arr)

        return np.exp(total_loss) / Y.size(0)


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
