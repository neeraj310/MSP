from tinyml.layers import Linear, ReLu
from tinyml.learner import Learner
from tinyml.losses import mse_loss
from tinyml.net import Sequential
from tinyml.optims import SGDOptimizer


class FullyConnectedNetwork():
    def __init__(self, num_fc_layers, num_neurons, activations, lr=0.05) -> None:
        self.num_fc_layers = num_fc_layers
        self.num_neurons = num_neurons
        self.activations = activations
        self.model = Sequential([])
        self.lr = lr
        self._build()
        self.model.summary()

    def _build(self):
        for idx in range(self.num_fc_layers):
            self.model.add(
                Linear('fc_{}'.format(idx),
                       self.num_neurons[idx],
                       self.num_neurons[idx+1])
            )
            if self.activations[idx] == 'relu':
                self.model.add(ReLu('relu_'.format(idx)))

    def fit(self, X, y, epochs=10, batch_size=10) -> None:
        learner = Learner(self.model, mse_loss, SGDOptimizer(lr=self.lr))
        learner.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)