from abc import ABC, abstractmethod

class BaseTopKMetric(ABC):
    def __init__(self, top_k = (1,), metric_name = None):
        self.top_k = top_k
        self.metric_name = metric_name
    
    @abstractmethod
    def __call__(self, outputs, targets):
        pass