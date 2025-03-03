import copy
from agent_information import AgentInformation
from cao import CAO
import numpy as np

class CentralizedDecisionMaking:
    def __init__(self, nr, D, dt, initP, fw, Tmax, NoPert, monomials, maxOrder,
                 pertrubUpBound, pertrubLowBound, perMono, testbed):
        self.n = nr
        self.D = D
        self.dt = dt
        self.pold = copy.deepcopy(initP)
        self.fw = fw
        self.testbed = testbed
        self.petrubConst = [0.0] * Tmax
        UpBound = pertrubUpBound * dt
        LowBound = pertrubLowBound * dt
        self.petrubConst[0] = UpBound
        for i in range(1, Tmax):
            self.petrubConst[i] = self.petrubConst[i-1] - (UpBound - LowBound) / Tmax
        self.AllAgents = []
        for i in range(nr):
            agent = AgentInformation(D)
            agent.setCurrentState(initP[i])
            self.AllAgents.append(agent)
        self.optimizer = CAO(self.AllAgents, D, NoPert, monomials, maxOrder, perMono, testbed)

    def produceDecisionVariables(self, t, p, J):
        # Convert p to a numpy array and make a copy.
        p = np.array(p, dtype=np.float64)
        hp = p.copy()
        if t <= self.fw:
            for i, agent in enumerate(self.AllAgents):
                agent.setPhistory(hp[i].tolist())
                ptemp = p.copy()
                ptemp[i] = self.pold[i]  # assuming self.pold[i] is already a copy
                jprevI = self.testbed.EvaluateCF(ptemp.tolist(), i)
                ja = (2 * J - jprevI) if t == 0 else (agent.getCostFunctionHistory()[-1] + J - jprevI)
                agent.setCostFunctionHistory(ja)
        else:
            for i, agent in enumerate(self.AllAgents):
                agent.removeFromStartSetPhistory(hp[i].tolist())
                ptemp = p.copy()
                ptemp[i] = self.pold[i]
                jprevI = self.testbed.EvaluateCF(ptemp.tolist(), i)
                ja = agent.getCostFunctionHistory()[self.fw] + J - jprevI
                agent.removeFromStartSetCostFunctionHistory(ja)
        self.pold = p.copy()
        pnew = []
        for i in range(self.n):
            new_action = self.optimizer.FindBestNewActions(i, self.petrubConst[t])
            self.AllAgents[i].setCurrentState(new_action)
            pnew.append(new_action)
        return pnew

    def getPetrubConst(self, t):
        return self.petrubConst[t]
