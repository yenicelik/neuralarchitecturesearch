
fixed_arc="0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"

class ChildNetwork:

    def __init__(self):
        """
            Here, we initialize the child network.
            From the DAG, we GENERATE a neural network.
            We USE SHARED WEIGHTS.
            We accept the X_train, Y_train to be able to TRAIN the model.
            We accept the X_valid, Y_valid to be able to VALIDATE the accuracy of the model.
            We FEED BACK the ACCURACY of the spawned model to the controller.
        """
        pass

    def _00_dag_to_nn(self):
        pass

    def _01_get_shared_weights(self):
        """
            We are interested in re-using weights if they have been saved before.
            As such, we can simply get the weights from memory
        :return:
        """
        pass

    def _02_train_nn(self):
        pass

    def _03_test_nn(self):
        pass



