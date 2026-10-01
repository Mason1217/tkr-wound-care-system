MODEL_REGISTRY = {}

def register_model(name):
    """
    Decorator used to registered model implementations.
    """
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered!")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator