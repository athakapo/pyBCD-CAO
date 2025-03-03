from abc import ABC, abstractmethod

class testbed_setup(ABC):
    def __init__(self):
        self.log_helper = None
        self.D = None          # dimension of decision vector
        self.nr = None         # number of robots
        self.dt = None
        self.last_known_decisions = None
        self.initial_decisions = None
        self.properties = {}
        self.noIter = None
        self.current_iter = 0  # will be set externally

    @abstractmethod
    def CalculateCF(self, decision_variables):
        pass

    @abstractmethod
    def EvaluateCF(self, decision_variables, r):
        pass

    @abstractmethod
    def isThisAValidDecisionCommand(self, r, decision_variables):
        pass

    @abstractmethod
    def fetchDecisionVector(self):
        pass

    @abstractmethod
    def worldConstructor(self):
        pass

    @abstractmethod
    def initializeLiveVisualization(self):
        pass

    @abstractmethod
    def updateLiveVisualization(self, iter, J):
        pass

    @abstractmethod
    def finalizeLiveVisualization(self):
        pass

    def setWriter(self, writer):
        self.log_helper = writer

    def getInitialDecisionVector(self):
        return self.initial_decisions

    def setInitialDecisionVector(self, initial_decisions):
        if len(initial_decisions) != self.nr:
            print("Error: mismatch in number of robots.")
            exit(1)
        elif len(initial_decisions[0]) != self.D:
            print("Error: mismatch in decision vector dimension (d).")
            exit(1)
        else:
            self.initial_decisions = initial_decisions

    def getLatestDecisionVariables(self):
        return self.last_known_decisions

    def PassProperties(self, props):
        self.properties = props
        self.dt = float(props["dt"])
        self.D = int(props["d"])
        self.nr = int(props["noRobots"])
        self.noIter = int(props["noIter"])
        if props.get("randomID", "false").lower() == "true":
            self.constructRandomInitialDecisions(float(props["minDimen"]), float(props["maxDimen"]))
        else:
            self.fetchDecisionVector()

    def constructRandomInitialDecisions(self, minD, maxD):
        import random
        valid = False
        while not valid:
            self.initial_decisions = []
            for i in range(self.nr):
                vec = [minD + (maxD - minD) * random.random() for _ in range(self.D)]
                self.initial_decisions.append(vec)
            self.updateAugmentedDecisionVector(self.initial_decisions)
            valid = True
            for i in range(self.nr):
                if not self.isThisAValidDecisionCommand(i, self.initial_decisions[i]):
                    valid = False
                    break

    def updateAugmentedDecisionVector(self, A):
        import copy
        self.last_known_decisions = copy.deepcopy(A)
