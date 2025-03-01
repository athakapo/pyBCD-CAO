class AgentInformation:
    def __init__(self, d):
        self.cost_function_history = []   # list of CF values over time
        self.phistory = []                # decision history (list of decision vectors)
        self.d = d
        self.currentState = [0.0] * d
        self.theta = None
        self.fringe = set()
        self.preservedFeatures = []
        self.usedFeatures = []

    def setusedFeatures(self, UF):
        self.usedFeatures = UF

    def getusedFeatures(self):
        return self.usedFeatures

    def setPreservedFeatures(self, pF):
        self.preservedFeatures = pF

    def getPreservedFeatures(self):
        return self.preservedFeatures

    def setFringe(self, f):
        self.fringe = f

    def getFringe(self):
        return self.fringe

    def setTheta(self, M):
        self.theta = M

    def getTheta(self):
        return self.theta

    def getCurrentState(self):
        return self.currentState

    def setCurrentState(self, v):
        self.currentState = v[:]  # make a copy

    def getCostFunctionHistory(self):
        return self.cost_function_history

    def setCostFunctionHistory(self, v):
        self.cost_function_history.append(v)

    def removeFromStartSetCostFunctionHistory(self, v):
        if self.cost_function_history:
            self.cost_function_history.pop(0)
        self.cost_function_history.append(v)

    def getPhistory(self):
        return self.phistory

    def setPhistory(self, v):
        self.phistory.append(v)

    def removeFromStartSetPhistory(self, v):
        if self.phistory:
            self.phistory.pop(0)
        self.phistory.append(v)
