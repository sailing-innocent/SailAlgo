def openai_client(endpoint, api_key):
    """
    Decorator to wrap OpenAI API calls with endpoint and API key.
    
    Args:
        endpoint (str): The API endpoint.
        api_key (str): The API key.
        
    Returns:
        function: Decorated function that takes prompt and model as arguments.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(endpoint, api_key, *args, **kwargs)
        return wrapper
    return decorator

