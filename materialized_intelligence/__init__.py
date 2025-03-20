from .sdk import MaterializedIntelligence

# Create an instance of the class
_instance = MaterializedIntelligence()

# Import all methods from the instance into the package namespace
from types import MethodType

for attr in dir(_instance):
    if callable(getattr(_instance, attr)) and not attr.startswith("__"):
        globals()[attr] = MethodType(getattr(_instance, attr).__func__, _instance)

# Clean up namespace
del MethodType, attr
