import math
import numpy as np
from testbed_setup import testbed_setup  # Adjust the import to match your package structure
from numba import njit

class Framework(testbed_setup):
    def __init__(self):
        super().__init__()
        self.pointsToMonitor = None
        self.kw = None
        self.n_p = None
        self.logHelper = None
        self.maxD = 1.0
        self.minD = 0.0
        self.thres = 0.0

    def worldConstructor(self):
        self.n_p = 2000
        self.kw = self.n_p / 25
        # Create an array of shape (n_p, 2) to store the monitoring points
        self.pointsToMonitor = np.zeros((self.n_p, 2))
        yTarget = 0.0
        for i in range(self.n_p):
            self.pointsToMonitor[i, 0] = 4 * (yTarget ** 4) - 12 * (yTarget ** 3) + 10 * (yTarget ** 2) - 2 * yTarget + 0.55
            self.pointsToMonitor[i, 1] = yTarget
            yTarget += 1.0 / self.n_p

    def isThisAValidDecisionCommand(self, r_i, decisionVariables):
        # Check if each decision variable is within [minD, maxD]
        for val in decisionVariables:
            if val > self.maxD or val < self.minD:
                return False

        # For each robot other than r_i, check that the new command is not too close
        latest = self.getLatestDecisionVariables()  # Expected to return an array-like of shape (nr, D)
        for r, current in enumerate(latest):
            if r != r_i:
                if Framework.EuclideanDist2D(np.asarray(current), np.asarray(decisionVariables)) < self.thres:
                    return False
        return True

    def CalculateCF(self, loc):
        return self.coreCalculation(loc)

    def EvaluateCF(self, loc, r_i):
        return self.coreCalculation(loc)

    def fetchDecisionVector(self):
        # Manually set initial positions (ensure the 'randomID' parameter is false)
        self.setInitialDecisionVector(np.array([
            [0.1, 0.1],
            [0.15, 0.3],
            [0.7, 0.01],
            [0.2, 0.18]
        ]))

    def setWriter(self, W):
        self.logHelper = W
        # Write the monitoring points to file using the log helper
        self.logHelper.WriteToFile(self.pointsToMonitor, "/pointsToMonitor.txt")

    def coreCalculation(self, loc):
        """
        Vectorized computation of the cost function for the HoldTheLine testbed.
        'loc' is a numpy array of shape (n_r, 2) containing the current robot positions.

        Returns:
          J: the cost function value.
        """
        # Ensure loc is a NumPy array of shape (n_r, 2)
        L = np.array(loc, dtype=np.float64)  # (n_r, 2)
        # P is the precomputed monitoring points from worldConstructor, shape: (n_p, 2)
        P = self.pointsToMonitor  # (n_p, 2)

        # Compute differences and distances using broadcasting:
        # diff: shape (n_p, n_r, 2)
        diff = P[:, None, :] - L[None, :, :]
        # dist: shape (n_p, n_r)
        dist = np.linalg.norm(diff, axis=2)

        # For each monitoring point, get the minimum distance and the robot index
        min_dist = np.min(dist, axis=1)  # shape: (n_p,)
        assigned = np.argmin(dist, axis=1)  # shape: (n_p,)

        # Initial cost: sum of all minimum distances
        J = np.sum(min_dist)

        n_r = L.shape[0]
        # For each robot, compute the minimal distance from any monitoring point (all points)
        PerRobotMinDis = np.min(dist, axis=0)  # shape: (n_r,)

        # Count how many points are assigned to each robot
        PointAssignedToRobot = np.bincount(assigned, minlength=n_r)

        # For any robot with no point assigned, add a penalty term
        for r in range(n_r):
            if PointAssignedToRobot[r] == 0:
                J += self.kw * PerRobotMinDis[r]

        return J

    @staticmethod
    @njit
    def EuclideanDist2D(a, b):
        diff = a - b
        return np.sqrt(np.sum(diff * diff))

    # This method is provided as an example and is not used in coreCalculation.
    @njit
    def CalculateDistanceLatLon(self, lat1, lat2, lon1, lon2, el1, el2):
        R = 6371  # Radius of the earth in kilometers
        latDistance = math.radians(lat2 - lat1)
        lonDistance = math.radians(lon2 - lon1)
        a = math.sin(latDistance / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(lonDistance / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c * 1000  # convert to meters
        height = el1 - el2
        distance = distance ** 2 + height ** 2
        return math.sqrt(distance)


    def initializeLiveVisualization(self):
        pass

    def updateLiveVisualization(self, iter, J):
        pass

    def finalizeLiveVisualization(self):
        pass
