import copy
from resource_loader import ResourceLoader
import importlib
import numpy as np

class OptimizationLoop:
    def __init__(self, testbedName, propertiesFILE=None, initialDecisions=None):
        self.JJ = []
        rl = ResourceLoader()
        if propertiesFILE is None:
            path = "testbeds/{}/Parameters.properties".format(testbedName)
            propertiesFILE = rl.get_properties_ap(path)
        self.propertiesFILE = propertiesFILE
        self.testbedName = testbedName

        # Dynamically load the appropriate testbed framework
        if testbedName == "HoldTheLine":
            mod = importlib.import_module("testbeds.HoldTheLine.Framework")
            testbed_class = getattr(mod, "Framework")
        elif testbedName == "AdaptiveCoverage2D":
            mod = importlib.import_module("testbeds.AdaptiveCoverage2D.Framework")
            testbed_class = getattr(mod, "AdaptiveCoverage2DFramework")
        else:
            raise ValueError("Unknown testbed name")
        self.testbed = testbed_class()
        self.testbed.PassProperties(propertiesFILE)
        self.testbed.worldConstructor()
        if initialDecisions is not None:
            self.testbed.setInitialDecisionVector(initialDecisions)
            self.IDecisionsPass = True
            self.initialDecisions = initialDecisions
        else:
            self.IDecisionsPass = False

        Tmax = int(propertiesFILE["noIter"])
        self.testbed.noIter = Tmax  # pass Tmax for CF calculation
        # Create planner
        from centralized_decision_making import CentralizedDecisionMaking
        noRobots = int(propertiesFILE["noRobots"])
        d = int(propertiesFILE["d"])
        dt = float(propertiesFILE["dt"])
        if self.testbed.getInitialDecisionVector() is None:
            initial_vec = self.testbed.last_known_decisions
        else:
            initial_vec = self.testbed.getInitialDecisionVector()
        tw = int(propertiesFILE["tw"])
        noPerturbations = int(propertiesFILE["noPerturbations"])
        monomials = int(propertiesFILE["noMonomials"])
        maxOrder = int(propertiesFILE["monoMaxOrder"])
        pertrubConstMax = float(propertiesFILE["pertrubConstMax"])
        pertrubConstMin = float(propertiesFILE["pertrubConstMin"])
        perMono = float(propertiesFILE["perMono"])
        self.planner = CentralizedDecisionMaking(noRobots, d, dt, initial_vec, tw, Tmax,
                                                  noPerturbations, monomials, maxOrder,
                                                  pertrubConstMax, pertrubConstMin, perMono, self.testbed)
        decisionVariable = np.array(self.testbed.getInitialDecisionVector(), dtype=np.float64).copy()
        for iter in range(Tmax):
            self.testbed.current_iter = iter
            self.testbed.updateAugmentedDecisionVector(decisionVariable)
            J = self.testbed.CalculateCF(decisionVariable)
            self.JJ.append(J)
            if self.propertiesFILE.get("displayTime&CF", "false").lower() == "true":
                print("Testbed: {} | Timestamp: {} , CF: {}".format(testbedName, iter, J))
            decisionVariable = self.planner.produceDecisionVariables(iter, decisionVariable, J)

    def getJJ(self):
        return self.JJ
