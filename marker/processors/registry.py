# Processor plugin registry for Marker

PROCESSOR_REGISTRY = {}

def register_processor(name):
    def decorator(cls):
        PROCESSOR_REGISTRY[name] = cls
        return cls
    return decorator

def get_processor(name):
    return PROCESSOR_REGISTRY.get(name)
