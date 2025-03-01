import sys
from optimization_loop import OptimizationLoop

if __name__ == '__main__':
    if len(sys.argv) == 2:
        testbed = sys.argv[1]
        OptimizationLoop(testbed)
    else:
        print("Re-run the application having as argument ONLY the name of the testbed (e.g. 'HoldTheLine')")
