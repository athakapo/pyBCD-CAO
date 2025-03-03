from helpers.resource_loader import ResourceLoader
import importlib
import numpy as np

class OptimizationLoop:
    def __init__(self, testbed, propertiesFILE=None, initialDecisions=None):
        self.JJ = []
        self.testbed = testbed
        if propertiesFILE.get("enableVisualization", "false").lower() == "true":
            self.testbed.initializeLiveVisualization()
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
            if propertiesFILE.get("enableVisualization", "false").lower() == "true":
                self.testbed.updateLiveVisualization(iter, J)
            self.JJ.append(J)
            if propertiesFILE.get("displayTime&CF", "false").lower() == "true":
                print("Testbed: {} | Timestamp: {} , CF: {}".format(testbed.testbed_name, iter, J))
            decisionVariable = self.planner.produceDecisionVariables(iter, decisionVariable, J)
        if propertiesFILE.get("enableVisualization", "false").lower() == "true":
            self.testbed.finalizeLiveVisualization()
    def getJJ(self):
        return self.JJ
