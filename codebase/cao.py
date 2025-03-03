import numpy as np
import random
import os
from concurrent.futures import ThreadPoolExecutor
from numba import njit
from numba import prange


def global_eval_candidate(args):
    """
    Evaluate one candidate decision.

    Parameters:
      args: tuple (r, prob, cur_state, candidate, calc_estimation_fn)
        - r: index of the robot.
        - prob: the problem instance (must be picklable) that provides isThisAValidDecisionCommand.
        - cur_state: the current state for the robot (e.g., the AgentInformation instance for r).
        - candidate: a NumPy array representing a candidate decision vector.
        - calc_estimation_fn: a function that computes the candidate's estimated cost.

    Returns:
      (cost, candidate): cost is a float; candidate is the NumPy array.
    """
    r, prob, cur_state, candidate, calc_estimation_fn = args
    # Check candidate validity (convert candidate to list for the validity check)
    if not prob.isThisAValidDecisionCommand(r, candidate.tolist()):
        return float('inf'), candidate
    # Compute the estimated cost
    cost = calc_estimation_fn(cur_state, candidate.tolist())
    return cost, candidate

class CAO:
    def __init__(self, AgentsInf, d, noPerturbations, noMonomials, monoMaxOrder, perMono, prob):
        self.AgentsInf = AgentsInf
        self.d = d
        self.noPerturbations = noPerturbations
        self.noMonomials = noMonomials
        self.monoMaxOrder = monoMaxOrder
        self.n_ss = d
        self.prob = prob
        self.perMono = perMono
        self.rnd = random.Random()
        self.CalculateMonomialsPerOrder()

    def CalculateMonomialsPerOrder(self):
        # In the Java code these are hard-coded.
        self.monomPerOrder = [2, 3, 4]
        self.preservedMonomials = [2, 3, 4]

    def FindBestNewActions(self, r, CurrentPetrub):
        """
        A vectorized and parallelized candidate evaluation using ProcessPoolExecutor.
        """
        # Get current decision vector from agent r
        CurRobot = self.AgentsInf[r]
        # Update features via polynomial estimator (updates theta, etc.)
        self.polynomialEstimator(CurRobot)

        # Base candidate as a numpy array
        perturbBase = np.array(CurRobot.getCurrentState(), dtype=np.float64)

        # Number of candidates: include the unperturbed candidate.
        num_candidates = self.noPerturbations + 1

        # Generate candidates in batch:
        candidates = np.empty((num_candidates, self.d), dtype=np.float64)
        candidates[0] = perturbBase
        # For the rest, generate random numbers in [-1,1] and scale
        random_perturbations = 2 * np.random.random((num_candidates - 1, self.d)) - 1
        candidates[1:] = perturbBase + CurrentPetrub * random_perturbations

        # Build argument list for global_eval_candidate.
        # Note: self.calculateEstimation is an instance method. For ProcessPoolExecutor it must be picklable.
        # If needed, you can refactor calculateEstimation as a static method or ensure it doesn't capture non-picklable state.
        args_list = []
        for cand in candidates:
            args_list.append((r, self.prob, CurRobot, cand, self.calculateEstimation))

        best_cost = float('inf')
        best_candidate = candidates[0]

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(global_eval_candidate, args_list, chunksize=1000))

        for cost, cand in results:
            if cost < best_cost:
                best_cost = cost
                best_candidate = cand

        return best_candidate.tolist()

    def polynomialEstimator(self, system):
        """
        Compute a polynomial estimator based on the agent’s history.
        This version converts the decision history and feature lists into NumPy arrays,
        then calls the Numba–compiled helper build_phi to construct phi.
        """
        self.CalculateCurrentFeatures(system)
        m_history = len(system.getCostFunctionHistory())
        # Convert decision history (phistory) to a 2D NumPy array of shape (m_history, D)
        SystemM = np.array(system.getPhistory(), dtype=np.float64)

        # Convert used features (a list of lists) to a tuple of NumPy arrays.
        # Each element corresponds to a monomial order and is of shape (order+1, n_features)
        features_list = system.getusedFeatures()
        features_tuple = tuple(np.array(f, dtype=np.int64) for f in features_list)

        total_rows = self.noMonomials + 1
        # Build phi using the Numba-compiled helper
        phi = self.build_phi(SystemM, features_tuple, self.monoMaxOrder, total_rows)

        # Build the cost table (as a 2D NumPy array)
        CostTable = np.array(system.getCostFunctionHistory(), dtype=np.float64).reshape(m_history, 1)

        try:
            Theta = np.linalg.pinv(phi.T).dot(CostTable)
            system.setTheta(Theta)
        except Exception as e:
            print("Error computing pseudo-inverse:", e)
        self.CalculateDominantFeatures(system)

    @staticmethod
    @njit(nogil=True, parallel=True)
    def build_phi(SystemM, features_tuple, monoMaxOrder, total_rows):
        m_history = SystemM.shape[0]
        phi = np.zeros((total_rows, m_history))
        # Parallelize over the history entries
        for j in prange(m_history):
            q = 0
            for k in range(monoMaxOrder):
                feats = features_tuple[k]
                n_features = feats.shape[1]
                for ii in range(n_features):
                    prod = 1.0
                    for i in range(k + 1):
                        prod *= SystemM[j, feats[i, ii]]
                    phi[q, j] = prod
                    q += 1
            phi[q, j] = 1.0
        return phi

    def CalculateCurrentFeatures(self, system):
        """
        Generate new features (monomials) based on the agent’s decision history.
        For each order (from 1 to monoMaxOrder), determine how many new features
        to generate, then create them by randomly selecting indices.
        """
        usedFeatures = []
        R_fringe = system.getFringe()  # expected to be a set
        for i in range(self.monoMaxOrder):
            if len(system.getPreservedFeatures()) > i:
                NoPrev = len(system.getPreservedFeatures()[i][0])
            else:
                NoPrev = 0
            NoFet = self.monomPerOrder[i] - NoPrev
            j = 0
            newFetToADD = [[None for _ in range(NoFet)] for _ in range(i+1)]
            while j < NoFet:
                lfet = []
                for k in range(i+1):
                    lfet.append(self.rnd.randint(0, self.n_ss - 1))
                lfet.sort()
                lkey = str(lfet)
                if lkey not in R_fringe:
                    R_fringe.add(lkey)
                    for k in range(i+1):
                        newFetToADD[k][j] = lfet[k]
                    j += 1
            cols = NoPrev + NoFet
            finalFeat = [[None for _ in range(cols)] for _ in range(i+1)]
            if NoPrev > 0:
                for q in range(i+1):
                    for k in range(NoPrev):
                        finalFeat[q][k] = system.getPreservedFeatures()[i][q][k]
            for q in range(i+1):
                k1 = 0
                for k in range(NoPrev, cols):
                    finalFeat[q][k] = newFetToADD[q][k1]
                    k1 += 1
            usedFeatures.append(finalFeat)
        system.setusedFeatures(usedFeatures)

    def CalculateDominantFeatures(self, system):
        """
        From the generated features, select the dominant ones based on the current
        theta values. For each order, if there are more features than allowed by
        preservedMonomials, sort the features by descending absolute theta value and
        retain the top ones.
        """
        R_fringe = set()
        dominantFeatures = []
        indxS = 0
        for i in range(self.monoMaxOrder):
            usedFeat = system.getusedFeatures()[i]  # shape: (i+1) x (# features)
            num_features = len(usedFeat[0])
            indxE = indxS + num_features - 1
            if (indxE - indxS + 1) > self.preservedMonomials[i]:
                pairTheta = []
                for j in range(indxE - indxS):
                    thetaElem = system.getTheta()[indxS + j, 0]
                    pairTheta.append((j, abs(thetaElem)))
                pairTheta.sort(key=lambda x: x[1], reverse=True)
                currFeature = [[None for _ in range(self.preservedMonomials[i])] for _ in range(i+1)]
                for j in range(self.preservedMonomials[i]):
                    NextValuableFeature = pairTheta[j][0]
                    tempkey = []
                    for k in range(i+1):
                        vi = usedFeat[k][NextValuableFeature]
                        currFeature[k][j] = vi
                        tempkey.append(vi)
                    R_fringe.add(str(tempkey))
                dominantFeatures.append(currFeature)
            else:
                dominantFeatures.append(usedFeat)
            indxS = indxE + 1
        system.setFringe(R_fringe)
        system.setPreservedFeatures(dominantFeatures)

    def calculateEstimation(self, system, cand):
        """
            Given a candidate decision vector 'cand', convert the feature data to numpy arrays
            and compute the estimated cost using the fast_calculate_estimation helper.
            """
        # Convert the used features into a tuple of numpy arrays.
        # Each element corresponds to a monomial order and is of shape (order+1, n_features)
        features_list = system.getusedFeatures()  # originally a list of lists
        features_tuple = tuple(np.array(f, dtype=np.int64) for f in features_list)

        theta = system.getTheta()  # assuming this is already a numpy array of shape (total_rows, 1)
        total_rows = self.noMonomials + 1  # same as in your original code
        return self.fast_calculate_estimation(theta, features_tuple, np.array(cand, dtype=np.float64), self.monoMaxOrder,
                                              total_rows)

    @staticmethod
    @njit(nogil=True)
    def fast_calculate_estimation(theta, features_tuple, cand, monoMaxOrder, total_rows):
        PHI = np.empty((total_rows, 1), dtype=np.float64)
        q = 0
        for k in range(monoMaxOrder):
            feats = features_tuple[k]
            n_features = feats.shape[1]
            for ii in range(n_features):
                prod = 1.0
                for i in range(k + 1):
                    prod *= cand[feats[i, ii]]
                PHI[q, 0] = prod
                q += 1
        PHI[q, 0] = 1.0
        s = 0.0
        for i in range(total_rows):
            s += theta[i, 0] * PHI[i, 0]
        return s
