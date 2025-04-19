class NaNDetectedException(Exception):
    """Exception raised when NaN values are detected in tensors."""
    pass

def check_nan(tensor, name):
    """Check if tensor contains NaN values and raise an exception if found."""
    if torch.isnan(tensor).any():
        raise NaNDetectedException(f"NaN values detected in {name}")
    return tensor