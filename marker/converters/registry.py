# Converter plugin registry for Marker

CONVERTER_REGISTRY = {}

def register_converter(name):
    def decorator(cls):
        CONVERTER_REGISTRY[name] = cls
        return cls
    return decorator

def get_converter(name):
    return CONVERTER_REGISTRY.get(name)
