import inspect

def filter_valid_args(callable_obj, all_args: dict) -> dict:
    """
    Automatically filters out arguments that `callable_obj` (function or class) does not accept.
    
    Logic:
    1. If the target accepts **kwargs, return all_args without filtering.
    2. If the target does not accept **kwargs, only retain arguments defined in the signature.
    """
    # Retrieve the signature
    try:
        sig = inspect.signature(callable_obj)
    except ValueError:
        # Handle certain built-in functions where the signature cannot be retrieved
        return all_args

    params = sig.parameters

    # Check for **kwargs (VAR_KEYWORD)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    if has_var_keyword:
        return all_args

    # Get all valid parameter names
    valid_keys = set(params.keys())
    
    # Filter arguments
    filtered_args = {
        k: v for k, v in all_args.items() 
        if k in valid_keys
    }
    
    return filtered_args