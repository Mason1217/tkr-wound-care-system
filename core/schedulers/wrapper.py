from torch.optim.lr_scheduler import ReduceLROnPlateau

class SchedulerWrapper:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.use_metric = isinstance(scheduler, ReduceLROnPlateau)
    
    def step(self, metric=None):
        if self.use_metric:
            if metric is None:
                print(f"Warning: {self.scheduler.__class__.__name__} requires metric, but None provided.")
                return 
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
            
    def state_dict(self):
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)