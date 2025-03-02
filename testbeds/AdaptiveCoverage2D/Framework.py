import numpy as np
from math import sqrt, pow

from numba import njit
from scipy.stats import multivariate_normal
from testbed_setup import testbed_setup


import numpy as np
from numba import njit

@njit(nogil=True)
def centroid_approx_numeric(V,Q,p,allCalKappa,hatAC,RealA,Kappa,nr,N,m):
    """
    A purely numeric version of the CentroidApproximation logic.

    Parameters
    ----------
    V : int32[:]
        1D array of Voronoi region indices for each world point (length N).
    Q : float64[:, :]
        2D array (N x 2) of world points.
    p : float64[:, :]
        2D array (nr x 2) of robot positions.
    allCalKappa : float64[:, :]
        2D array (N x m), where row j is calKappa for Q[j].
    hatAC : float64[:, :]
        2D array (m x 1) of current mixing estimate.
    RealA : float64[:, :]
        2D array (m x 1) of real mixing parameters (used in original code).
    Kappa : float64[:, :]
        2D array (2 x 2) for your gain matrix (self.Kappa).
    nr : int
        Number of robots.
    N : int
        Number of world points.
    m : int
        Number of Gaussian components or monomials.

    Returns
    -------
    FvalOut : float64[:, :, :]
        3D array (nr x m x m) that corresponds to your Fvallocal (one (m x m) matrix per robot).
    Clocal : float64[:, :]
        2D array (nr x 2) for the centroids per robot.
    H_val : float64
        A scalar cost value (the sum of norms, etc.) / 2.0.
    """

    # Allocate accumulators
    F1Arr = np.zeros((nr, m, 2), dtype=np.float64)
    F2Arr = np.zeros((nr, 2, m), dtype=np.float64)
    MArr = np.zeros(nr, dtype=np.float64)
    LVec = np.zeros((nr, 2), dtype=np.float64)

    H_val_accum = 0.0  # We'll divide by 2.0 at the end

    # -- MAIN LOOP over each world point --
    for j in range(N):
        r = V[j]  # which robot "owns" point j
        calKappa = allCalKappa[j]  # shape (m,)

        # Compute hatPhi and realPhi by dotting calKappa with hatAC and RealA
        hatPhi = 0.0
        realPhi = 0.0
        for i in range(m):
            hatPhi += calKappa[i] * hatAC[i, 0]
            realPhi += calKappa[i] * RealA[i, 0]

        dx = Q[j, 0] - p[r, 0]
        dy = Q[j, 1] - p[r, 1]

        # Update F1Arr[r], shape (m,2)
        # We'll do a small loop over i in range(m)
        for i in range(m):
            F1Arr[r, i, 0] += calKappa[i] * dx
            F1Arr[r, i, 1] += calKappa[i] * dy

        # Update F2Arr[r], shape (2,m)
        for i in range(m):
            F2Arr[r, 0, i] += dx * calKappa[i]
            F2Arr[r, 1, i] += dy * calKappa[i]

        MArr[r] += hatPhi
        LVec[r, 0] += Q[j, 0] * hatPhi
        LVec[r, 1] += Q[j, 1] * hatPhi

        # Norm of the point2robot vector
        dist = (dx * dx + dy * dy) ** 0.5
        H_val_accum += dist * hatPhi

    # Now build FvalOut (nr, m, m) and Clocal (nr,2)
    FvalOut = np.zeros((nr, m, m), dtype=np.float64)
    Clocal = np.zeros((nr, 2), dtype=np.float64)

    for r2 in range(nr):
        Mval = MArr[r2]
        if Mval != 0.0:
            # partial = F1Arr[r2] @ Kappa  => shape (m,2)
            partial = np.zeros((m, 2), dtype=np.float64)
            for i in range(m):
                px = F1Arr[r2, i, 0]
                py = F1Arr[r2, i, 1]
                # (m,2)*(2,2) => we do each column
                # partial[i,0] = px*Kappa[0,0] + py*Kappa[1,0]
                partial[i, 0] = px * Kappa[0, 0] + py * Kappa[1, 0]
                partial[i, 1] = px * Kappa[0, 1] + py * Kappa[1, 1]

            # full = partial @ F2Arr[r2] => shape(m,m)
            # partial is (m,2), F2Arr[r2] is (2,m)
            # We'll do manual loop to avoid function calls in nopython mode
            for i in range(m):
                for j2 in range(m):
                    # partial(i,0)*F2Arr(r2,0,j2) + partial(i,1)*F2Arr(r2,1,j2)
                    FvalOut[r2, i, j2] = (
                        partial[i, 0] * F2Arr[r2, 0, j2]
                        + partial[i, 1] * F2Arr[r2, 1, j2]
                    )
                    FvalOut[r2, i, j2] *= 1.0 / Mval

            # Clocal[r2] = LVec[r2] / Mval
            Clocal[r2, 0] = LVec[r2, 0] / Mval
            Clocal[r2, 1] = LVec[r2, 1] / Mval
        else:
            # If Mval == 0, it stays all zeros
            pass

    return FvalOut, Clocal, H_val_accum / 2.0


class AdaptiveCoverage2DFramework(testbed_setup):
    def __init__(self):
        super().__init__()
        # Constants (same as in Java)
        self.sigmaj = 0.02
        self.amin = 0.1
        self.g = 0.01
        self.z = 0.005
        self.maxD = 1.0
        self.minD = 0.0

    def worldConstructor(self):
        # Number of points constituting the world [Resolution]
        self.N = 225  # 225 (or 529 if desired; must be a perfect square)
        self.sl = int(sqrt(self.N))
        # 2D Representation of the world
        self.Q = self.equalSeparation(self.sl)
        # Covariances on gaussians
        self.sigma = np.array([[self.sigmaj, 0], [0, self.sigmaj]])
        # Number of gaussians in the same row
        self.sm = 7
        self.m = int(pow(self.sm, 2))
        self.gaussianC = self.equalSeparation(self.sm)
        # Estimation Parameters
        self.hatA = []
        for i in range(self.nr):
            self.hatA.append(self.InitializeSimpleMatrix(self.m, 1, self.amin))
        self.Lamda = []
        self.l = []
        for i in range(self.nr):
            # new SimpleMatrix(m, m) in Java gives a zero matrix
            self.Lamda.append(self.InitializeSimpleMatrix(self.m, self.m, 0.0))
            self.l.append(self.InitializeSimpleMatrix(self.m, 1, 0.0))
        # Gains
        self.Kappa = np.identity(2) * 3.0
        self.Gamma = np.identity(self.m)
        # Centralized parameters
        self.w = self.InitializeSimpleMatrix(self.nr, 1, 1.0)
        self.lamdaC = self.InitializeSimpleMatrix(self.m, self.m, 0.0)
        self.lC = self.InitializeSimpleMatrix(self.m, 1, 0.0)
        self.hatAC = self.InitializeSimpleMatrix(self.m, 1, self.amin)
        # AllCalKappa: precompute for each point in the world
        self.AllCalKappa = []
        for i in range(self.N):
            self.AllCalKappa.append(self.CalKappa(self.Q[i]))
        # True mixing parameters
        self.RealA = self.InitializeSimpleMatrix(self.m, 1, self.amin)
        self.RealA[0, 0] = 100.0
        if self.m > 15:
            self.RealA[15, 0] = 100.0

    def CalculateCF(self, p):
        # Calculate Voronoi partitions and cost function value,
        # then update the estimation of the mixing vector.
        V = self.VoronoiPartitions(updateParam=True, p=p)
        J = self.CentroidApproximation(updateParam=True, V=V, p=p)
        self.UpdateHatA()
        return J

    def EvaluateCF(self, p, r):
        # Evaluate CF without updating parameters
        V = self.VoronoiPartitions(updateParam=False, p=p)
        return self.CentroidApproximation(updateParam=False, V=V, p=p)

    def isThisAValidDecisionCommand(self, i, p):
        # Check if every component is within [minD, maxD]
        for j in range(self.D):
            if p[j] > self.maxD or p[j] < self.minD:
                return False
        return True

    def fetchDecisionVector(self):
        # Set the robots’ initial positions manually.
        self.setInitialDecisionVector(np.array([
            [0.748782920935759, 0.523484207016743],
            [0.521725763522798, 0.662285767439358],
            [0.618314982927528, 0.850496608951951],
            [0.987035204212623, 0.683118335623724],
            [0.560976459754866, 0.480022865952500],
            [0.796815587856705, 0.712148754079348],
            [0.904113237958921, 0.006839213657844],
            [0.687208306090933, 0.641243548188644],
            [0.822574509070901, 0.141788922472766],
            [0.863995313984828, 0.247451873545336]
        ]))

    def VoronoiPartitions(self, updateParam, p):
        # Convert p to a NumPy array if not already
        p = np.array(p)  # shape: (nr, 2)
        # Vectorized distance calculation between each world point and robot position
        diff = self.Q[:, np.newaxis, :] - p[np.newaxis, :, :]  # shape: (N, nr, 2)
        distances = np.sum(diff ** 2, axis=2)  # squared Euclidean distance
        V = np.argmin(distances, axis=1)

        if updateParam:
            A = np.reshape(V, (self.sl, self.sl))
            localLap = np.identity(self.nr)
            for i in range(self.sl):
                for j in range(self.sl):
                    if (i + 1 < self.sl) and (j - 1 >= 0):
                        if A[i + 1, j - 1] != A[i, j]:
                            localLap[A[i + 1, j - 1], A[i, j]] = 1.0
                            localLap[A[i, j], A[i + 1, j - 1]] = 1.0
                    if i + 1 < self.sl:
                        if A[i + 1, j] != A[i, j]:
                            localLap[A[i + 1, j], A[i, j]] = 1.0
                            localLap[A[i, j], A[i + 1, j]] = 1.0
                    if (i + 1 < self.sl) and (j + 1 < self.sl):
                        if A[i + 1, j + 1] != A[i, j]:
                            localLap[A[i + 1, j + 1], A[i, j]] = 1.0
                            localLap[A[i, j], A[i + 1, j + 1]] = 1.0
                    if j + 1 < self.sl:
                        if A[i, j + 1] != A[i, j]:
                            localLap[A[i, j + 1], A[i, j]] = 1.0
                            localLap[A[i, j], A[i, j + 1]] = 1.0
            self.Lap = localLap
            self.LapPerRobot = []
            for i in range(self.nr):
                LapR = np.zeros((1, self.nr))
                for r in range(self.nr):
                    LapR[0, r] = self.Lap[i, r]
                self.LapPerRobot.append(LapR)
        return V


    def CentroidApproximation(self, updateParam, V, p):
        """
        The high-level method that orchestrates the centroid computation.
        Calls the Numba-compiled helper for the numeric logic, then if updateParam
        is True, updates class attributes (Fval, C, etc.).
        """
        # Ensure V is a NumPy integer array
        V_np = np.array(V, dtype=np.int32)

        # Convert p to a NumPy array if it's not already
        p_arr = np.array(p, dtype=np.float64)

        # Prepare a 2D array for allCalKappa if not already stored that way:
        # Suppose self.AllCalKappa is a list of shape (N,) each of which is (m,1).
        # We'll build an (N x m) array:
        allCalKappa = np.zeros((self.N, self.m), dtype=np.float64)
        for j in range(self.N):
            # self.AllCalKappa[j] is shape (m,1). Flatten to (m,)
            allCalKappa[j, :] = self.AllCalKappa[j].ravel()

        # Now call the numeric helper
        FvalOut, Clocal, H_val = centroid_approx_numeric(
            V_np,
            self.Q,          # shape (N, 2)
            p_arr,           # shape (nr, 2)
            allCalKappa,     # shape (N, m)
            self.hatAC,      # shape (m,1)
            self.RealA,      # shape (m,1)
            self.Kappa,      # shape (2,2)
            self.nr,
            self.N,
            self.m
        )

        # If updateParam is True, store the numeric outputs in self
        if updateParam:
            # Convert the FvalOut (nr, m, m) back into a Python list if your code expects that
            Fvallocal = []
            for r in range(self.nr):
                # FvalOut[r] is (m, m), store as a single element in Fvallocal
                # or adapt as needed
                Fvallocal.append(FvalOut[r])

            self.Fval = Fvallocal    # was originally a list of length nr
            self.C = Clocal         # shape (nr, 2)

        return H_val

    def UpdateHatA(self):
        w = self.InitializeSimpleMatrix(self.nr, 1, 1.0)
        hatAT = [self.InitializeSimpleMatrix(self.m, 1, 0.0) for _ in range(self.nr)]
        for r in range(self.nr):
            # Use the latest decision for robot r; assume getLatestDecisionVariables() is implemented in Setup
            calcKapppa = self.CalKappa(self.getLatestDecisionVariables()[r])
            lamdaRobot = np.dot(calcKapppa, calcKapppa.T) * (w[r, 0] * self.dt)
            lRobot = np.dot(calcKapppa, np.dot(calcKapppa.T, self.RealA)) * (w[r, 0] * self.dt)
            self.lamdaC = self.lamdaC + lamdaRobot
            self.lC = self.lC + lRobot
            self.Lamda[r] = self.Lamda[r] + lamdaRobot
            self.l[r] = self.l[r] + lRobot
            cons = self.InitializeSimpleMatrix(self.m, 1, 0.0)
            if self.z != 0:
                temp = np.dot(self.InitializeSimpleMatrix(self.nr, 1, 1.0), self.LapPerRobot[r]) + self.Lap
                w = self.countPerRow(temp, 2.0)
                for j in range(self.nr):
                    if r != j:
                        cons = cons + (self.hatA[r] - self.hatA[j]) * w[j, 0]
            dotApre = (np.dot(self.Fval[r], self.hatA[r]) * -1.0) - ((np.dot(self.Lamda[r], self.hatA[r]) - self.l[r]) * self.g) - (cons * self.z)
            hatAT[r] = self.hatA[r] + np.dot(self.Gamma, dotApre) * self.dt
            Iproj = self.CalIproj(hatAT[r], dotApre)
            self.hatA[r] = self.hatA[r] + np.dot(self.Gamma, self.dt * (dotApre - np.dot(Iproj, dotApre)))
        dotApreC = (np.dot(self.lamdaC, self.hatAC) - self.lC) * (-self.g)
        Iproj = self.CalIproj(self.hatAC + np.dot(self.Gamma, dotApreC) * self.dt, dotApreC)
        self.hatAC = self.hatAC + np.dot(self.Gamma, self.dt * (dotApreC - np.dot(Iproj, dotApreC)))

    def norm(self, A, B):
        return np.linalg.norm(np.array(A) - np.array(B))

    def equalSeparation(self, q):
        A = np.zeros((q * q, 2))
        k = 0
        for i in range(1, q + 1):
            for j in range(1, q + 1):
                A[k, 0] = i / (q + 1.0)
                A[k, 1] = j / (q + 1.0)
                k += 1
        return A

    def CalKappa(self, q):
        # Updated to exactly match the Java computation.
        # In Java, the density is computed with covariance sigma = [[sigmaj, 0], [0, sigmaj]]
        # which yields: 1/(2π * sigmaj) * exp(-0.5 * ||q-mu||^2 / sigmaj)
        v = self.sigma[0, 0]  # sigmaj (variance value, as used in the Java code)
        norm_const = 1.0 / (2 * np.pi * v)
        diff = q - self.gaussianC  # vectorized; shape: (m,2)
        exponent = -0.5 * np.sum(diff**2, axis=1) / v
        pdf_vals = norm_const * np.exp(exponent)
        return pdf_vals.reshape(self.m, 1)

    def InitializeSimpleMatrix(self, rows, cols, value):
        return np.full((rows, cols), value, dtype=float)

    def countPerRow(self, A, num):
        # Count occurrences of num in each row of matrix A and return as a column vector.
        counts = np.sum(A == num, axis=1, keepdims=True)
        return counts.astype(float)

    def CalIproj(self, ha, dotha):
        Iproj = np.identity(self.m)
        for i in range(self.m):
            if (ha[i, 0] > self.amin) or (ha[i, 0] == self.amin and dotha[i, 0] >= 0):
                Iproj[i, i] = 0.0
        return Iproj

    def getLatestDecisionVariables(self):
        # This method is assumed to be implemented in Setup.
        # It should return the latest decision variables (a 2D array where each row corresponds to a robot).
        return self.last_known_decisions

    def initializeLiveVisualization(self):
        pass

    def updateLiveVisualization(self, iter, J):
        pass

    def finalizeLiveVisualization(self):
        pass
