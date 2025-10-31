from .sdk import Sutro

# Create a singleton instance
_instance = Sutro()

# Export all public methods from the instance
for attr in dir(_instance):
    if callable(getattr(_instance, attr)) and not attr.startswith("_"):
        globals()[attr] = getattr(_instance, attr)

# Optionally export the class itself if users need direct access
# Sutro is already imported and available

# Define __all__ for clean imports
__all__ = ["Sutro"] + [
    attr
    for attr in dir(_instance)
    if callable(getattr(_instance, attr)) and not attr.startswith("_")
]

# Clean up namespace
del attr
