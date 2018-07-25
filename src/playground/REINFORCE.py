import numpy as np

def take_action(cur_state, cur_action):
    """

    :param cur_state:
    :param cur_action: action that we take in the current state
    :return: The new state sn and the reward that we get by taking action `cur_action` in state `cur_state`
    """
    return cur_state * cur_action


def train_episode():

    policies = [
        (0, 1),
        (2, 0),
        (5, 3),
        (2, 2),
        (1, 6)
    ]

    theta = 0.
    alpha = 0.5

    # We sample an episode from the policy \pi_\theta

    for i in range(len(total_episodes)):

        history = take_actions()

    while next_episode:

    for i in range(len(policies)):
        cur_state = policies[i][0]
        cur_action = policies[i][1]

        r = take_action(cur_state, cur_action)

        for t in range(i):
            print("Going through all past policies!")
            print()

            gradient = np.random.rand(1)

            theta = theta + alpha * gradient

            print("New theta value is: ", theta)


if __name__ == "__main__":
    print("Trying out the REINFORCE algorithm")

    train_episode()


