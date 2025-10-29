from .sdk import Sutro

# Create an instance of the class
_instance = Sutro()

# Import all methods from the instance into the package namespace
from types import MethodType

for attr in dir(_instance):
    if attr.startswith("__"):
        continue
    value = getattr(_instance, attr)
    if isinstance(value, type):
        continue
    # If it's a bound method (has __func__), rebind; otherwise export the value directly
    if callable(value) and hasattr(value, "__func__"):
        globals()[attr] = MethodType(value.__func__, _instance)
    else:
        globals()[attr] = value

# Clean up namespace
del MethodType, attr
try:
    del value
except NameError:
    pass
