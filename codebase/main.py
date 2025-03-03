# main.py (simplified example)
import sys

from helpers.discover_testbeds import discover_testbeds
from helpers.resource_loader import ResourceLoader
from optimization_loop import OptimizationLoop

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py <TestbedName>")
        sys.exit(1)

    testbedName = sys.argv[1]

    testbeds_map = discover_testbeds()

    # See if user-supplied testbedName is in the discovered map
    if testbedName not in testbeds_map:
        print(f"Unknown testbed name: {testbedName}")
        print("Available testbeds:", list(testbeds_map.keys()))
        sys.exit(1)

    rl = ResourceLoader()
    path = "testbeds/{}/Parameters.properties".format(testbedName)
    propertiesFILE = rl.get_properties_ap(path)

    testbed_class = testbeds_map[testbedName]
    # Instantiate
    testbed_instance = testbed_class()

    # Kick off the optimization loop
    OptimizationLoop(testbed_instance, propertiesFILE)
