#!/usr/bin/env python3
import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Use TkAgg backend for visualization
import matplotlib.pyplot as plt
from testbed_setup import testbed_setup


class HazardEnclosureTestbed(testbed_setup):
    """
    HazardEnclosureTestbed simulates a collaborative multi-robot task:
    the robots must collectively form an enclosure around a dynamic hazardous region.

    Environment:
      - A circular hazard whose center and radius vary over time.
      - The desired team formation is a barrier located at a fixed margin outside the hazard.

    Global Objective:
      - Each robot should position itself near the desired barrier (i.e., its distance from the hazard center should match a target value).
      - The robots should be uniformly distributed around the hazard, minimizing large gaps in the enclosure.

    The cost function is global and is composed of:
      1. A barrier formation cost: the sum of squared errors between each robot's distance from the hazard center and the desired barrier radius.
      2. A gap penalty: a term penalizing the maximum angular gap between consecutive robots (sorted by angle around the hazard).

    This testbed provides additional live visualization functions (initializeLiveVisualization,
    updateLiveVisualization, finalizeLiveVisualization) that are optional and only used if the centralized
    decision-making algorithm wishes to display the evolving state.
    """

    def __init__(self):
        super().__init__()
        # Load parameters from file relative to this script.
        params_path = os.path.join(os.path.dirname(__file__), "Parameters.properties")
        self.load_parameters(params_path)

        # Global parameters (environment dimensions, number of robots, etc.)
        self.noRobots = int(self.getParameter("noRobots", 10))
        self.dt = float(self.getParameter("dt", 0.01))
        self.noIter = int(self.getParameter("noIter", 800))
        self.minDimen = float(self.getParameter("minDimen", 0.0))
        self.maxDimen = float(self.getParameter("maxDimen", 1.0))
        self.d = int(self.getParameter("d", 2))
        # For simulation, we still define a world (for visualization purposes)
        self.N = int(self.getParameter("N", 225))  # We use a grid to show the environment

        # Set the base class attributes required by testbed_setup.
        self.nr = self.noRobots
        self.D = self.d

        # Parameters for the hazard (dynamic region)
        self.hazard_margin = float(self.getParameter("hazard_margin", 0.1))  # desired gap from hazard to barrier
        self.lambda_gap = float(self.getParameter("lambda_gap", 1.0))  # weight for gap penalty

        # Dynamic hazard parameters (center and radius)
        # These may be tuned or scheduled; here we use simple sinusoidal functions.
        self.hazard_center = np.array([0.5, 0.5])
        self.hazard_radius = float(self.getParameter("hazard_radius", 0.15))
        self.desired_radius = self.hazard_radius + self.hazard_margin

        # Parameters to update the hazard over time
        self.hazard_center_amp = float(self.getParameter("hazard_center_amp", 0.2))
        self.hazard_center_speed = float(self.getParameter("hazard_center_speed", 0.05))
        self.hazard_radius_amp = float(self.getParameter("hazard_radius_amp", 0.05))
        self.hazard_radius_speed = float(self.getParameter("hazard_radius_speed", 0.03))

        # For visualization of the environment, we still build a base world.
        self.base_Q = None
        self.Q = None
        self.W = None  # not used in cost here, but for visualization
        self.worldConstructor()

        # Decision vectors (positions)
        self.initial_decisions = None
        self.last_known_decisions = None
        # We do not call fetchDecisionVector automatically here,
        # because we assume the initial positions will be provided externally.

        self.current_iter = 0  # updated externally

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
        Also update last_known_decisions.
        """
        self.initial_decisions = p
        self.last_known_decisions = p

    def getLatestDecisionVariables(self):
        """Return the most recent decision vector."""
        return self.last_known_decisions

    def isThisAValidDecisionCommand(self, i, p):
        """Check that each component of robot i's decision is within bounds."""
        for j in range(self.d):
            if p[j] < self.minDimen or p[j] > self.maxDimen:
                return False
        return True

    def fetchDecisionVector(self):
        """
        Generate an initial decision vector (list-of-lists).
        (This can be used if no external initialization is provided.)
        """
        p = [[float(np.random.uniform(self.minDimen, self.maxDimen)) for _ in range(self.d)]
             for _ in range(self.noRobots)]
        self.setInitialDecisionVector(p)
        return p

    # ---------- World Construction (for visualization) ----------
    def worldConstructor(self):
        """Construct a uniform grid for visualization purposes."""
        sl = int(np.sqrt(self.N))
        self.base_Q = self.equalSeparation(sl)
        self.Q = np.copy(self.base_Q)
        self.W = np.ones(self.N)

    def equalSeparation(self, q):
        """Generate a grid of points in [0,1]x[0,1]."""
        total_points = q * q
        points = np.zeros((total_points, self.d))
        k = 0
        for i in range(1, q + 1):
            for j in range(1, q + 1):
                points[k, 0] = i / (q + 1.0)
                points[k, 1] = j / (q + 1.0)
                k += 1
        return points

    # ---------- Hazard Update ----------
    def updateHazard(self, t):
        """
        Update the hazard region over time.
        The hazard center moves sinusoidally and the hazard radius oscillates.
        """
        self.hazard_center = np.array([
            0.5 + self.hazard_center_amp * np.sin(self.hazard_center_speed * t),
            0.5 + self.hazard_center_amp * np.cos(self.hazard_center_speed * t)
        ])
        self.hazard_radius = 0.15 + self.hazard_radius_amp * np.sin(self.hazard_radius_speed * t)
        self.desired_radius = self.hazard_radius + self.hazard_margin

    # ---------- Cost Function (Global Collaborative Objective) ----------
    def CalculateCF(self, p):
        """
        Compute the cost function for the enclosure task given decision vector p (list-of-lists).

        Components:
          1. Barrier formation cost: sum of squared errors between each robot's distance from
             the hazard center and the desired barrier radius.
          2. Gap penalty: the maximum difference between consecutive robot angles (around the hazard center).
        """
        # Update hazard based on current iteration
        self.updateHazard(self.current_iter)

        p_arr = np.array(p, dtype=np.float64)
        # Compute barrier cost
        dists = np.linalg.norm(p_arr - self.hazard_center, axis=1)
        barrier_errors = dists - self.desired_radius
        cost_barrier = np.sum(barrier_errors ** 2)

        # Compute gap penalty: sort robots by angle around hazard_center
        angles = np.arctan2(p_arr[:, 1] - self.hazard_center[1], p_arr[:, 0] - self.hazard_center[0])
        angles = np.sort(angles)
        # Wrap-around gap: add 2*pi to the first angle
        gaps = np.diff(np.concatenate([angles, [angles[0] + 2 * np.pi]]))
        cost_gap = np.max(gaps)  # or sum squared deviations from (2*pi/noRobots)

        total_cost = cost_barrier + self.lambda_gap * cost_gap
        return total_cost

    def EvaluateCF(self, p, r):
        """
        Evaluate cost function without updating internal state.
        Here, the cost is global, so robot index r is not used.
        """
        return self.CalculateCF(p)

    # ---------- Live Visualization for External Calls ----------
    def initializeLiveVisualization(self):
        """
        Initialize figure, axes, and related objects for live visualization.
        Call this at the start of the optimization loop.
        """
        if hasattr(self, "_liveVis_initialized") and self._liveVis_initialized:
            return

        plt.ion()
        self._fig = plt.figure(figsize=(8, 6))
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._cbar_ax = self._fig.add_axes([0.92, 0.2, 0.02, 0.6])
        self._sc = self._ax.scatter([], [], c=[], cmap='viridis', s=30)
        self._cbar = self._fig.colorbar(self._sc, cax=self._cbar_ax, label='World Grid Value')

        self._cost_history = []
        self._trajectories = []
        # For each robot, initialize trajectory storage if last_known_decisions exists
        if self.last_known_decisions is not None:
            p0 = self.last_known_decisions
            for i in range(self.noRobots):
                self._trajectories.append([np.array(p0[i])])
        else:
            for i in range(self.noRobots):
                self._trajectories.append([])

        self._liveVis_initialized = True

    def updateLiveVisualization(self, iteration, cost):
        """
        Update the live visualization with the current state.
        Typically called after each iteration of the centralized decision-making.
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
        # For visualization, we plot the base world grid (optional) as background
        sc = self._ax.scatter(self.Q[:, 0], self.Q[:, 1], c=self.W, cmap='viridis', s=30)
        sc.set_clim(vmin=min(self.W), vmax=max(self.W))
        self._cbar.update_normal(sc)

        # Plot robots
        self._ax.scatter(p_arr[:, 0], p_arr[:, 1], c='blue', s=50, label='Robots')
        # Plot desired barrier circle and hazard region
        hazard_circle = plt.Circle(self.hazard_center, self.hazard_radius, color='red', fill=False, linestyle='--',
                                   label='Hazard')
        barrier_circle = plt.Circle(self.hazard_center, self.desired_radius, color='green', fill=False, linestyle='-',
                                    label='Desired Barrier')
        self._ax.add_artist(hazard_circle)
        self._ax.add_artist(barrier_circle)

        # For showing enclosure, sort robots by angle and draw connecting lines
        angles = np.arctan2(p_arr[:, 1] - self.hazard_center[1], p_arr[:, 0] - self.hazard_center[0])
        sort_idx = np.argsort(angles)
        sorted_positions = p_arr[sort_idx]
        self._ax.plot(np.append(sorted_positions[:, 0], sorted_positions[0, 0]),
                      np.append(sorted_positions[:, 1], sorted_positions[0, 1]),
                      'k-', linewidth=1, label='Enclosure')

        # Draw trajectories for each robot
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
        Finalize the visualization session and (optionally) plot cost history.
        """
        if not hasattr(self, "_liveVis_initialized") or not self._liveVis_initialized:
            return
        plt.ioff()
        plt.show()
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


# ---------------------- (Optional) Standalone Test Main ----------------------
if __name__ == "__main__":
    # For standalone testing, you can generate a random initial decision vector if needed.
    testbed = HazardEnclosureTestbed()
    # Optionally, you can set initial_decisions externally. For demonstration, we'll use fetchDecisionVector.
    if testbed.getLatestDecisionVariables() is None:
        testbed.setInitialDecisionVector(testbed.fetchDecisionVector())
    print("Initial decision vector:", testbed.getLatestDecisionVariables())
    # Run live visualization using a naive update (here we simulate random updates)
    plt.ion()
    for t in range(500):
        # For demonstration, perform a small random update to the positions.
        p_arr = np.array(testbed.getLatestDecisionVariables(), dtype=np.float64)
        update = np.random.uniform(-0.002, 0.002, p_arr.shape)
        new_positions = p_arr + update
        new_positions = np.clip(new_positions, testbed.minDimen, testbed.maxDimen)
        testbed.setInitialDecisionVector(new_positions.tolist())
        testbed.last_known_decisions = testbed.getLatestDecisionVariables()
        cost = testbed.CalculateCF(testbed.getLatestDecisionVariables())
        testbed.current_iter = t
        testbed.updateLiveVisualization(t, cost)
    testbed.finalizeLiveVisualization()
