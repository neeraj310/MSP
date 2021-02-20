from timeit import default_timer as timer

from sklearn import metrics
from src.indexing.models import BaseModel
from src.indexing.learning.fully_connected_network import FullyConnectedNetwork

class FCNModel(BaseModel):
    def __init__(self) -> None:
        super().__init__('Fully Connected Neural Network')
        self.model = FullyConnectedNetwork(2, [1,4,1], ['relu','relu'])
    
    def train(self, x_train, y_train, x_test, y_test):
        start_time = timer()
        self.model.fit(x_train, y_train)
        end_time = timer()
        y_hat = self.model.predict(x_test)
        mse = metrics.mean_squared_error(y_test, y_hat)
        return mse, end_time - start_time
    
    def predict(self, X):
        X = X.reshape((1))
        return self.model.predict(X)