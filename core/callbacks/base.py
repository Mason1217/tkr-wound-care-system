class BaseCallback:
    def on_train_start(self, trainer, **kwargs): pass
    def on_train_end(self, trainer, **kwargs): pass
    
    def on_epoch_start(self, trainer, epoch, **kwargs): pass
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs): pass
    
    def on_batch_end(self, trainer, loss, **kwargs): pass
    
    def on_test_start(self, trainer, **kwargs): pass
    def on_test_end(self, trainer, metrics, **kwargs): pass