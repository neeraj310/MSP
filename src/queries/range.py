from timeit import default_timer as timer
from src.indexing.models import BaseModel
from src.indexing.utilities.dataloaders import split_train_test
from src.queries import Query

class RangeQuery(Query):
    def predict(self, upper_right, lower_left):
        pass

    def evaluate(self):
        pass