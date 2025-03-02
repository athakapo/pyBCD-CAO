#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
from testbed_setup import testbed_setup

class IntentAwareTestbed(testbed_setup):
    """
    IntentAware Testbed with a dynamic, non-uniform environment and a global, collaborative objective.

    inherited from: https://github.com/SiyuanQi-zz/intentMARL/tree/master

    Key Points:
      - The testbed does NOT internally update robot positions.
      - A dynamic environment is simulated by a moving hotspot, random perturbations, etc.
      - The cost function includes coverage, cohesion, and target-tracking terms.
      - We provide new methods to visualize each iteration from an external script (like centralized_decision_making.py).
    """

    def __init__(self):
        super().__init__()
        # Load parameters
        params_path = os.path.join(os.path.dirname(__file__), "Parameters.properties")
        self.load_parameters(params_path)

        # Global problem parameters
        self.noRobots = int(self.getParameter("noRobots", 10))
        self.dt = float(self.getParameter("dt", 0.01))
        self.noIter = int(self.getParameter("noIter", 800))
        self.minDimen = float(self.getParameter("minDimen", 0.0))
        self.maxDimen = float(self.getParameter("maxDimen", 1.0))
        self.d = int(self.getParameter("d", 2))
        self.N = int(self.getParameter("N", 225))  # Number of world points

        # Base class attributes
        self.nr = self.noRobots
        self.D = self.d

        # Additional parameters for global objectives
        self.lambda_cohesion = float(self.getParameter("lambda_cohesion", 1.0))
        self.lambda_target = float(self.getParameter("lambda_target", 1.0))
        self.noise_sigma = float(self.getParameter("noise_sigma", 0.005))

        # Dynamic environment: base grid + weights
        self.base_Q = None
        self.Q = None
        self.W = None
        self.target_amplitude = float(self.getParameter("target_amplitude", 0.2))
        self.target_speed = float(self.getParameter("target_speed", 0.1))

        # Intent parameters
        self.intent_alpha = float(self.getParameter("intent_alpha", 0.1))
        self.intent_beta = float(self.getParameter("intent_beta", 0.5))
        self.intent = np.zeros((self.noRobots, self.d))

        # Decision vectors
        self.initial_decisions = None
        self.last_known_decisions = None

        # Build the world
        self.worldConstructor()
        self.current_iter = 0  # Will be updated externally

        # Initialize dynamic environment
        self.updateWorld(self.current_iter)

        # -- Remove the random re-initialization call --
        # self.initial_decisions = self.fetchDecisionVector()
        # self.last_known_decisions = self.initial_decisions

        # Visualization objects (for live updates)
        self._liveVis_initialized = False
        self._cost_history = []
        self._fig = None
        self._ax = None
        self._cbar_ax = None
        self._sc = None
        self._cbar = None
        self._trajectories = []

    # ---------- Parameter and Utility Functions ----------
    def load_parameters(self, filename):
        """Simple parser for a Properties file."""
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
        """Retrieve a parameter value with a default if not found."""
        return self.properties.get(key, default)

    def setInitialDecisionVector(self, p):
        """
        Store the decision vector (list-of-lists).
        Also update last_known_decisions so the testbed uses these positions.
        """
        self.initial_decisions = p
        self.last_known_decisions = p

    def getLatestDecisionVariables(self):
        """Return the most recent decision vector."""
        return self.last_known_decisions

    def isThisAValidDecisionCommand(self, i, p):
        """Check that each component of robot i's decision is within [minDimen, maxDimen]."""
        for j in range(self.d):
            if p[j] < self.minDimen or p[j] > self.maxDimen:
                return False
        return True

    def fetchDecisionVector(self):
        """
        Generate a random initial decision vector (list-of-lists).
        You can call this manually if you want random spawn,
        but it's no longer auto-called in the constructor.
        """
        p = [[float(np.random.uniform(self.minDimen, self.maxDimen)) for _ in range(self.d)]
             for _ in range(self.noRobots)]
        self.setInitialDecisionVector(p)
        return p

    # ---------- World Construction & Dynamics ----------
    def worldConstructor(self):
        """Build a uniform grid as the base world."""
        sl = int(np.sqrt(self.N))
        self.base_Q = self.equalSeparation(sl)
        self.Q = np.copy(self.base_Q)
        self.W = np.ones(self.N)

    def equalSeparation(self, q):
        """Generate a uniform grid of points in [0,1] x [0,1]."""
        total_points = q * q
        points = np.zeros((total_points, self.d))
        k = 0
        for i in range(1, q + 1):
            for j in range(1, q + 1):
                points[k, 0] = i / (q + 1.0)
                points[k, 1] = j / (q + 1.0)
                k += 1
        return points

    def updateWorld(self, t):
        """
        Update world points (self.Q) and weights (self.W) for a dynamic, non-uniform environment.
        """
        hotspot_center = np.array([0.5 + 0.3 * np.sin(self.target_speed * t),
                                   0.5 + 0.3 * np.cos(self.target_speed * t)])
        A = 3.0
        sigma_hot = 0.1

        noise_world = np.random.normal(0, 0.005, self.base_Q.shape)
        self.Q = self.base_Q + noise_world

        diff = self.Q - hotspot_center
        d2 = np.sum(diff ** 2, axis=1)
        self.W = 1.0 + A * np.exp(-d2 / (2 * sigma_hot ** 2))

    # ---------- Voronoi and Centroid Computations ----------
    def VoronoiPartitions(self, p):
        diff = self.Q[:, np.newaxis, :] - p[np.newaxis, :, :]
        distances = np.sum(diff ** 2, axis=2)
        V = np.argmin(distances, axis=1)
        return V

    def computeCentroids(self, V, p):
        centroids = np.zeros((self.noRobots, self.d))
        counts = np.zeros(self.noRobots)
        for i in range(len(V)):
            centroids[V[i]] += self.Q[i]
            counts[V[i]] += 1
        for r in range(self.noRobots):
            if counts[r] > 0:
                centroids[r] /= counts[r]
            else:
                centroids[r] = p[r]
        return centroids

    # ---------- Cost Function (Coverage + Cohesion + Target) ----------
    def CalculateCF(self, p):
        """
        Compute overall cost for the given decision vector p (list-of-lists).
        Updates dynamic world based on self.current_iter.
        """
        self.updateWorld(self.current_iter)
        p_arr = np.array(p, dtype=np.float64)
        V = self.VoronoiPartitions(p_arr)

        # Weighted coverage cost (with sensor noise)
        coverage_cost = 0.0
        for i in range(self.N):
            noise = np.random.normal(0, self.noise_sigma)
            dist = np.linalg.norm(self.Q[i] - p_arr[V[i]]) + noise
            coverage_cost += self.W[i] * abs(dist)
        coverage_cost /= 2.0

        # Cohesion penalty
        team_center = np.mean(p_arr, axis=0)
        cohesion_penalty = np.sum(np.linalg.norm(p_arr - team_center, axis=1) ** 2)

        # Target tracking penalty
        target = np.array([
            0.5 + self.target_amplitude * np.sin(self.target_speed * self.current_iter),
            0.5 + self.target_amplitude * np.cos(self.target_speed * self.current_iter)
        ])
        target_penalty = np.linalg.norm(team_center - target) ** 2

        total_cost = coverage_cost + self.lambda_cohesion * cohesion_penalty + self.lambda_target * target_penalty
        return total_cost

    def EvaluateCF(self, p, r):
        """
        Evaluate cost function without updating state.
        p is a list-of-lists, r is robot index (unused here).
        """
        p_arr = np.array(p, dtype=np.float64)
        V = self.VoronoiPartitions(p_arr)
        coverage_cost = 0.0
        for i in range(self.N):
            noise = np.random.normal(0, self.noise_sigma)
            dist = np.linalg.norm(self.Q[i] - p_arr[V[i]]) + noise
            coverage_cost += self.W[i] * abs(dist)
        coverage_cost /= 2.0

        team_center = np.mean(p_arr, axis=0)
        cohesion_penalty = np.sum(np.linalg.norm(p_arr - team_center, axis=1) ** 2)
        target = np.array([
            0.5 + self.target_amplitude * np.sin(self.target_speed * self.current_iter),
            0.5 + self.target_amplitude * np.cos(self.target_speed * self.current_iter)
        ])
        target_penalty = np.linalg.norm(team_center - target) ** 2

        return coverage_cost + self.lambda_cohesion * cohesion_penalty + self.lambda_target * target_penalty

    # ---------- Live Visualization for External Calls ----------
    def initializeLiveVisualization(self):
        """
        Initialize figure, axes, colorbar, and other structures for live updates.
        Call this once (e.g. at the start of the optimization loop).
        """
        if self._liveVis_initialized:
            return  # already initialized

        plt.ion()
        self._fig = plt.figure(figsize=(8, 6))
        self._ax = self._fig.add_subplot(1, 1, 1)
        # Colorbar axis on the right side
        self._cbar_ax = self._fig.add_axes([0.92, 0.2, 0.02, 0.6])

        # Create a dummy scatter to attach a colorbar
        self._sc = self._ax.scatter([], [], c=[], cmap='viridis', s=30)
        self._cbar = self._fig.colorbar(self._sc, cax=self._cbar_ax, label='Importance Weight')

        # Initialize cost history, trajectories, etc.
        self._cost_history = []
        self._trajectories = []
        # For each robot, store a list of positions for drawing trails
        if self.last_known_decisions is not None:
            p = self.last_known_decisions
            for i in range(self.noRobots):
                self._trajectories.append([np.array(p[i])])
        else:
            for i in range(self.noRobots):
                self._trajectories.append([])

        self._liveVis_initialized = True

    def updateLiveVisualization(self, iteration, cost):
        """
        Update the plot with the current world points, weights, robot positions, etc.
        This is typically called after each iteration in centralized_decision_making.

        iteration: current iteration index
        cost: the computed cost at this iteration
        """
        if not self._liveVis_initialized:
            self.initializeLiveVisualization()

        self.current_iter = iteration
        p = self.last_known_decisions
        if p is None:
            return

        p_arr = np.array(p, dtype=np.float64)

        # Update cost history
        self._cost_history.append(cost)

        # Update trajectory data
        for i in range(self.noRobots):
            self._trajectories[i].append(p_arr[i].copy())

        # Clear main axis
        self._ax.clear()

        # Re-scatter world points with updated importance weights
        self._sc = self._ax.scatter(self.Q[:, 0], self.Q[:, 1], c=self.W, cmap='viridis', s=30)
        # Update color limits
        self._sc.set_clim(vmin=min(self.W), vmax=max(self.W))
        # Update colorbar to match new scatter
        self._cbar.update_normal(self._sc)

        # Plot robots, centroids, team center, target
        V = self.VoronoiPartitions(p_arr)
        centroids = self.computeCentroids(V, p_arr)
        team_center = np.mean(p_arr, axis=0)
        target = np.array([
            0.5 + self.target_amplitude * np.sin(self.target_speed * iteration),
            0.5 + self.target_amplitude * np.cos(self.target_speed * iteration)
        ])

        self._ax.scatter(p_arr[:, 0], p_arr[:, 1], c='blue', s=50, label='Robots')
        self._ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=50, label='Centroids')
        self._ax.scatter(team_center[0], team_center[1], c='green', marker='*', s=100, label='Team Center')
        self._ax.scatter(target[0], target[1], c='magenta', marker='o', s=100, label='Moving Target')

        # Draw lines from each robot to its centroid
        for i in range(self.noRobots):
            self._ax.plot([p_arr[i, 0], centroids[i, 0]],
                          [p_arr[i, 1], centroids[i, 1]], 'k--', linewidth=0.5)

        # Draw robot trajectories
        for i in range(self.noRobots):
            traj_arr = np.array(self._trajectories[i])
            self._ax.plot(traj_arr[:, 0], traj_arr[:, 1], 'c-', linewidth=1)

        self._ax.set_title(f"Iteration {iteration} | Cost: {cost:.3f}")
        self._ax.set_xlim(self.minDimen, self.maxDimen)
        self._ax.set_ylim(self.minDimen, self.maxDimen)
        self._ax.legend(loc='upper right')

        plt.pause(0.01)

    def finalizeLiveVisualization(self):
        """
        Finalize the interactive session.
        Optionally, plot the cost history if desired.
        """
        if not self._liveVis_initialized:
            return
        plt.ioff()
        plt.show()

        # Plot cost history if we want
        if self._cost_history:
            plt.figure(figsize=(8, 4))
            plt.plot(self._cost_history, label='Cost Function')
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Cost Function over Iterations")
            plt.legend()
            plt.grid(True)
            plt.show()

        self._liveVis_initialized = False

# ---------------------- (Optional) No main function here ----------------------
