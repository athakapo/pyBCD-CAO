#!/usr/bin/env python3
import os
import numpy as np
import matplotlib

from testbeds.testbed_setup import testbed_setup

matplotlib.use('TkAgg')  # Use TkAgg for visualization
import matplotlib.pyplot as plt


class MultiRobotActiveExplorationTestbed(testbed_setup):
    """
    MultiRobotActiveExplorationTestbed implements a decentralized multi-robot active exploration problem.

    The objective is to collaboratively reduce the uncertainty in an unknown environment.
    This is achieved by minimizing the following global cost:

        J = sum_{x in G} w(x) * exp(-beta * min_{p in P} ||x - p||)

    where:
      - G is a discrete grid covering the operational area.
      - w(x) is an importance (weight) function defined over the area (e.g. uniform or spatially varying).
      - beta is a positive constant that controls the decay of uncertainty with distance.
      - P is the set of robot positions.

    This cost function is global, i.e. it does not decompose into per-robot objectives, and hence forces a collaborative strategy.
    """

    def __init__(self):
        super().__init__()
        # If the class has an attribute _testbed_subdir, copy it to the instance
        if hasattr(self.__class__, "_testbed_subdir"):
            self.testbed_name = self.__class__._testbed_subdir
        else:
            self.testbed_name = "UnknownTestbed"
        # Load parameters from file relative to this script.
        params_path = os.path.join(os.path.dirname(__file__), "Parameters.properties")
        self.load_parameters(params_path)

        # Operational area parameters.
        self.minDimen = float(self.getParameter("minDimen", 0.0))
        self.maxDimen = float(self.getParameter("maxDimen", 10.0))
        self.noRobots = int(self.getParameter("noRobots", 4))  # e.g., 4 robots
        self.d = int(self.getParameter("d", 2))
        self.dt = float(self.getParameter("dt", 0.05))
        self.noIter = int(self.getParameter("noIter", 800))
        # For numerical integration / visualization grid (should be a perfect square).
        self.N = int(self.getParameter("N", 400))

        # Set required attributes from testbed_setup.
        self.nr = self.noRobots
        self.D = self.d

        # Exploration parameters.
        # beta controls the decay of uncertainty with distance.
        self.beta = float(self.getParameter("beta", 1.0))
        # Importance weight function parameters: here we use a simple linear variation in y-direction.
        self.w_min = float(self.getParameter("w_min", 0.5))
        self.w_max = float(self.getParameter("w_max", 1.0))

        # Build a grid for the domain G (for cost computation and visualization).
        self.worldConstructor()

        # Decision vectors: these are set externally by the optimization loop.
        self.initial_decisions = None
        self.last_known_decisions = None
        self.current_iter = 0  # to be updated by the optimization loop

        # Visualization objects.
        self._liveVis_initialized = False
        self._cost_history = []
        self._fig = None
        self._ax = None
        self._trajectories = []

    # ---------------- Parameter and Utility Functions ----------------
    def load_parameters(self, filename):
        """Parse a Parameters.properties file."""
        self.properties = {}
        try:
            with open(filename, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        self.properties[key.strip()] = val.strip()
        except Exception as e:
            print(f"Error loading parameters from {filename}: {e}")

    def getParameter(self, key, default):
        return self.properties.get(key, default)

    def setInitialDecisionVector(self, p):
        """Store the decision vector and update the latest decisions."""
        self.initial_decisions = p
        self.last_known_decisions = p

    def getLatestDecisionVariables(self):
        return self.last_known_decisions

    def isThisAValidDecisionCommand(self, i, p):
        # Check that each coordinate is within [minDimen, maxDimen].
        for j in range(self.d):
            if p[j] < self.minDimen or p[j] > self.maxDimen:
                return False
        return True

    def fetchDecisionVector(self):
        """
        Generate an initial decision vector.
        For active exploration, robots can be initialized uniformly randomly in the domain.
        """
        p = []
        for i in range(self.noRobots):
            pos = [float(np.random.uniform(self.minDimen, self.maxDimen)) for _ in range(self.d)]
            p.append(pos)
        self.setInitialDecisionVector(p)
        return p

    # ---------------- World Construction ----------------
    def worldConstructor(self):
        """
        Construct a uniform grid covering the operational area.
        This grid (self.base_Q) is used for numerical integration of the cost function.
        """
        q = int(np.sqrt(self.N))
        total_points = q * q
        points = np.zeros((total_points, self.d))
        k = 0
        for i in range(1, q + 1):
            for j in range(1, q + 1):
                points[k, 0] = self.minDimen + i / (q + 1.0) * (self.maxDimen - self.minDimen)
                points[k, 1] = self.minDimen + j / (q + 1.0) * (self.maxDimen - self.minDimen)
                k += 1
        self.base_Q = points
        self.Q = np.copy(points)
        # The weight function will be computed later based on grid coordinates.
        self.W = np.ones(total_points)

    # ---------------- Global Cost Function ----------------
    def _computeGlobalCost(self, p_arr):
        """
        Compute the global active exploration cost:

          J = sum_{x in G} w(x) * exp( - beta * min_{p in P} ||x - p|| )

        where:
          - G is the grid (self.base_Q)
          - w(x) is an importance weight computed as a linear function in the y-coordinate:
                w(x) = w_min + (w_max - w_min) * ((y - minDimen) / (maxDimen - minDimen))
          - beta is a positive constant.
        """
        M = self.base_Q.shape[0]
        # Compute weights: for each grid point x, use its y-coordinate.
        Y = self.base_Q[:, 1]
        Y_scaled = (Y - self.minDimen) / (self.maxDimen - self.minDimen)
        weights = self.w_min + (self.w_max - self.w_min) * Y_scaled

        # For each grid point, compute distance to the nearest robot.
        diff_uav = self.base_Q[:, np.newaxis, :] - p_arr[np.newaxis, :, :]  # shape (M, n, 2)
        distances = np.linalg.norm(diff_uav, axis=2)  # shape (M, n)
        d_min = np.min(distances, axis=1)  # shape (M,)

        # Compute cost as sum over grid of weighted residual uncertainty.
        local_cost = weights * np.exp(- self.beta * d_min)
        total_cost = np.sum(local_cost)
        return total_cost

    def CalculateCF(self, p):
        """
        Calculate the global cost function (active exploration objective).
        Since the environment is assumed static in this version, no state update is performed.
        """
        p_arr = np.array(p, dtype=np.float64)
        return self._computeGlobalCost(p_arr)

    def EvaluateCF(self, p, r):
        """
        Evaluate the cost function without modifying any state.
        The argument 'r' is ignored since the cost is global.
        """
        p_arr = np.array(p, dtype=np.float64)
        return self._computeGlobalCost(p_arr)

    # ---------------- Live Visualization ----------------
    def initializeLiveVisualization(self):
        """
        Initialize live visualization.
        The plot shows the importance density (w(x)), the grid, and robot positions.
        """
        if hasattr(self, "_liveVis_initialized") and self._liveVis_initialized:
            return
        plt.ion()
        self._fig = plt.figure(figsize=(8, 6))
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._cost_history = []
        self._trajectories = []
        if self.last_known_decisions is not None:
            for i in range(self.noRobots):
                self._trajectories.append([np.array(self.last_known_decisions[i])])
        else:
            for i in range(self.noRobots):
                self._trajectories.append([])
        self._liveVis_initialized = True

    def updateLiveVisualization(self, iteration, cost):
        """
        Update the live visualization with current robot positions and exploration cost.
        """
        if not hasattr(self, "_liveVis_initialized") or not self._liveVis_initialized:
            self.initializeLiveVisualization()
        self.current_iter = iteration
        p = self.last_known_decisions
        if p is None:
            return
        p_arr = np.array(p, dtype=np.float64)
        self._cost_history.append(cost)
        for i in range(self.noRobots):
            self._trajectories[i].append(p_arr[i].copy())

        self._ax.clear()

        # Plot the importance (weight) field.
        q = int(np.sqrt(self.N))
        X = self.base_Q[:, 0].reshape(q, q)
        Y = self.base_Q[:, 1].reshape(q, q)
        Y_scaled = (Y - self.minDimen) / (self.maxDimen - self.minDimen)
        phi_grid = self.w_min + (self.w_max - self.w_min) * Y_scaled
        self._ax.imshow(phi_grid, extent=[self.minDimen, self.maxDimen, self.minDimen, self.maxDimen],
                        origin='lower', cmap='YlOrRd', alpha=0.6)

        # Plot robot positions.
        self._ax.scatter(p_arr[:, 0], p_arr[:, 1], c='blue', s=50, label='Robots')
        # Optionally plot trajectories.
        for i in range(self.noRobots):
            traj_arr = np.array(self._trajectories[i])
            if traj_arr.size > 0:
                self._ax.plot(traj_arr[:, 0], traj_arr[:, 1], 'c-', linewidth=1)
        self._ax.set_title(f"Iteration {iteration} | Exploration Cost: {cost:.3f}")
        self._ax.set_xlim(self.minDimen, self.maxDimen)
        self._ax.set_ylim(self.minDimen, self.maxDimen)
        self._ax.legend(loc='upper right')
        plt.pause(0.01)

    def finalizeLiveVisualization(self):
        """
        Finalize visualization and show cost evolution.
        """
        if not hasattr(self, "_liveVis_initialized") or not self._liveVis_initialized:
            return
        plt.ioff()
        plt.show()
        if self._cost_history:
            plt.figure(figsize=(8, 4))
            plt.plot(self._cost_history, label='Exploration Cost')
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Exploration Cost over Iterations")
            plt.legend()
            plt.grid(True)
            plt.show()
        self._liveVis_initialized = False

    # ---------------- Additional Methods ----------------
    def updateAugmentedDecisionVector(self, A):
        """
        Update the latest decisions (called by the optimization loop).
        """
        import copy
        self.last_known_decisions = copy.deepcopy(A)
