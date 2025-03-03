import os

class ResourceLoader:
    def get_properties_ap(self, resource_path):
        """Loads a .properties file and returns a dict of properties."""
        props = {}
        with open(resource_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    props[key.strip()] = value.strip()
        return props