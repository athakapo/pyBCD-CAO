import os
import importlib
import inspect
from testbeds.testbed_setup import testbed_setup


def discover_testbeds():
    """
    Dynamically discover all subdirectories under 'testbeds/' that contain a 'Framework.py'
    and a class inheriting from testbed_setup.

    Returns:
        A dict mapping subdirectory name (e.g. 'HoldTheLine') to the discovered class object.
    """
    # We need the path to the main repo folder that contains "testbeds/"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    testbeds_dir = os.path.join(base_dir, 'testbeds')
    available_testbeds = {}

    for entry in os.scandir(testbeds_dir):
        if entry.is_dir():
            framework_py = os.path.join(entry.path, "Framework.py")
            if os.path.isfile(framework_py):
                subdir_name = entry.name
                module_name = f"testbeds.{subdir_name}.Framework"
                try:
                    mod = importlib.import_module(module_name)
                except ImportError:
                    continue

                for name, obj in inspect.getmembers(mod, inspect.isclass):
                    if issubclass(obj, testbed_setup) and obj is not testbed_setup:
                        # Attach the subdir name to the class
                        setattr(obj, "_testbed_subdir", subdir_name)
                        available_testbeds[subdir_name] = obj
                        break

    return available_testbeds