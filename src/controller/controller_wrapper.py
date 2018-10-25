"""
    Wrapper class around the controller
"""
import torch
from torch.autograd import Variable

from src.config import C_DEVICE
from src.controller.controller_network import ControllerLSTM
from src.controller.debug_utils import example_reward
from src.model_config import ARG
from src.utils.debug_utils.tensorboard_tools import tx_writer


class ControllerWrapper:

    def __init__(self, controller_model: ControllerLSTM):
        """
            Includes common operations such as REINFORCE
        """

        self.controller_model = controller_model

        self.controller_optimizer = torch.optim.Adam(
            self.controller_model.parameters(),
            lr=ARG.controller_lr
        )

    def sample_dag(self):

        with torch.no_grad():
            dag_ops, entropy_history, log_probability_history = \
                self.controller_model.forward(input=None)

        return dag_ops

    def train_controller(self, reward_function):
        """
            Implements the REINFORCE algorithm
        :validation_loss_function:
            This function needs to take as input the dag.
        :return:
        """
        # The reward_baseline is the expected average value
        reward_baseline = None
        entropy_history = []
        adv_history = []
        reward_history = []
        loss_history = []

        # Gotta reset all the parameters here
        total_loss = 0.
        validation_idx = 0

        # We sample a few different architectures with the controller
        # and the the individual loss (without further training the weights
        for step in range(ARG.controller_max_step):

            # 1.st: sample the model
            dag_ops, entropy_history, log_probability_history = \
                self.controller_model.forward(input=None)

            # TODO: assume for now we don't use the entropy stuff

            # 2.nd: get the reward (by passing the validation data)
            reward = reward_function(dag_ops)

            print("Dag is. ", [x.item() for x in dag_ops])

            reward_history.append(reward)

            # Calculate the baseline
            if reward_baseline is None:
                reward_baseline = reward
            else:
                decay = ARG.ema_baseline_decay
                reward_baseline = decay * reward_baseline + (1 - decay) * reward

            # 3.rd: Calculate the policy loss
            adv = reward - reward_baseline
            adv_history.append(adv)

            # Need to convert log probability history
            # and entropy to torch variables!
            log_probabilities = torch.cat(log_probability_history)
            loss = -log_probabilities * Variable(
                torch.FloatTensor([adv])).to(C_DEVICE)  # TODO something fucked up with the formats here
            loss = loss.sum()  # Receiving the final loss
            loss_history.append(loss)

            # Add the values to the tensorboard!
            tx_writer.add_scalar('controller/validation_loss', loss, step)
            tx_writer.add_scalar('controller/reward', reward, step)
            tx_writer.add_scalar('controller/advantage', adv, step)

            # 4.th: Apply the backprop state to the controller weights
            self.controller_optimizer.zero_grad()
            loss.backward()
            self.controller_optimizer.step()

        print("Reward history: ", [x for x in reward_history])
        print("Adv history: ", [x for x in adv_history])
        print("Loss history: ", [x for x in loss_history])


if __name__ == "__main__":
    # Let's spawn the controller
    controller = ControllerLSTM()

    # let's use the example loss function:
    controller_wrapper = ControllerWrapper(controller)
    controller_wrapper.train_controller(
        reward_function=example_reward
    )
