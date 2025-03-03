#!/usr/bin/env python3
import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Use Matplotlib fallback for 2D visualization
import matplotlib.pyplot as plt
from testbed_setup import testbed_setup


class HazardEnclosureTestbed(testbed_setup):
    """
    HazardEnclosureTestbed simulates a collaborative multi-robot task:
    the robots must collectively form an enclosure around a dynamic hazardous region.

    Environment:
      - A circular hazard whose center and radius evolve over time according to an advection–diffusion model.
      - The desired team formation is a barrier located at a fixed margin outside the hazard.

    Global Objective:
      - The cost function is global and includes:
          1. Barrier formation cost: sum of squared errors between each robot's distance from the hazard center
             and the desired barrier radius.
          2. Uniform spacing cost: the sum of squared deviations of the angular gaps from the ideal gap,
             where the ideal gap is computed as:
                 ideal_gap = 2 * desired_radius * sin(pi / noRobots)

    New Requirements:
      - Operational area is larger (e.g., [0,10]×[0,10]).
      - Hazard is relatively small (base hazard radius ≈ 1.0).
      - The hazard moves more randomly: its center is updated via an advection–diffusion model.
      - Robots’ initial positions are generated in a subregion (lower left quadrant).
      - All parameters (including hazard dynamics and intensity parameters) are set via Parameters.properties.

    Visualization:
      - A 2D live visualization is provided that displays a background intensity field (Gaussian around the hazard),
        the hazard and desired barrier as circles, and the robot positions with their enclosure and trajectories.
      - The previous scatter plot of world grid dots and colorbar have been removed.
    """

    def __init__(self):
        super().__init__()
        # Load parameters from file relative to this script.
        params_path = os.path.join(os.path.dirname(__file__), "Parameters.properties")
        self.load_parameters(params_path)

        # Operational area
        self.minDimen = float(self.getParameter("minDimen", 0.0))
        self.maxDimen = float(self.getParameter("maxDimen", 10.0))
        self.noRobots = int(self.getParameter("noRobots", 10))
        self.d = int(self.getParameter("d", 2))
        self.dt = float(self.getParameter("dt", 0.01))
        self.noIter = int(self.getParameter("noIter", 800))
        self.N = int(self.getParameter("N", 225))  # For background grid (not used in cost)

        # Set base attributes required by testbed_setup.
        self.nr = self.noRobots
        self.D = self.d

        # Hazard parameters
        self.hazard_margin = float(self.getParameter("hazard_margin", 0.2))
        self.lambda_gap = float(self.getParameter("lambda_gap", 1.0))
        self.hazard_radius_base = float(self.getParameter("hazard_radius_base", 1.0))
        self.hazard_radius_amp = float(self.getParameter("hazard_radius_amp", 0.2))
        self.hazard_radius_speed = float(self.getParameter("hazard_radius_speed", 0.03))
        # Initialize hazard center at the center of the operational area
        self.hazard_center = np.array([(self.minDimen + self.maxDimen) / 2.0,
                                       (self.minDimen + self.maxDimen) / 2.0])
        self.desired_radius = self.hazard_radius_base + self.hazard_margin

        # Advection–diffusion parameters for hazard evolution
        self.wind_u = float(self.getParameter("wind_u", 0.1))
        self.wind_v = float(self.getParameter("wind_v", 0.0))
        self.diffusion_coefficient = float(self.getParameter("diffusion_coefficient", 0.01))
        # Additional hazard center dynamics (not used directly here)
        self.hazard_center_amp = float(self.getParameter("hazard_center_amp", 1.0))
        self.hazard_center_speed = float(self.getParameter("hazard_center_speed", 0.05))

        # Intensity parameter for visualization of hazard (Gaussian spread)
        self.hazard_sigma = float(self.getParameter("hazard_sigma", 0.5))

        # Build a base world grid for background visualization (this grid will not be drawn as dots)
        self.base_Q = None
        self.Q = None
        self.W = None
        self.worldConstructor()

        # Decision vectors: expected to be set externally.
        self.initial_decisions = None
        self.last_known_decisions = None

        self.current_iter = 0  # Updated externally

        # Visualization objects for live updates (Matplotlib fallback)
        self._liveVis_initialized = False
        self._cost_history = []
        self._fig = None
        self._ax = None
        self._trajectories = []

    # ---------- Parameter and Utility Functions ----------
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
        """Store the decision vector (list-of-lists) and update last_known_decisions."""
        self.initial_decisions = p
        self.last_known_decisions = p

    def getLatestDecisionVariables(self):
        return self.last_known_decisions

    def isThisAValidDecisionCommand(self, i, p):
        for j in range(self.d):
            if p[j] < self.minDimen or p[j] > self.maxDimen:
                return False
        return True

    def fetchDecisionVector(self):
        """
        Generate an initial decision vector.
        Robots are initialized in the lower left quadrant of the operational area.
        """
        lower_bound = self.minDimen
        upper_bound = (self.minDimen + self.maxDimen) / 2.0
        p = [[float(np.random.uniform(lower_bound, upper_bound)) for _ in range(self.d)]
             for _ in range(self.noRobots)]
        self.setInitialDecisionVector(p)
        return p

    # ---------- World Construction (for background visualization) ----------
    def worldConstructor(self):
        """Construct a uniform grid as the background for visualization."""
        sl = int(np.sqrt(self.N))
        self.base_Q = self.equalSeparation(sl)
        self.Q = np.copy(self.base_Q)
        self.W = np.ones(self.N)

    def equalSeparation(self, q):
        total_points = q * q
        points = np.zeros((total_points, self.d))
        k = 0
        for i in range(1, q + 1):
            for j in range(1, q + 1):
                # Scale grid points to cover the operational area
                points[k, 0] = i / (q + 1.0) * self.maxDimen
                points[k, 1] = j / (q + 1.0) * self.maxDimen
                k += 1
        return points

    # ---------- Hazard Update (Advection–Diffusion Model) ----------
    def updateHazard(self, t):
        """
        Update the hazard region using an advection–diffusion model.

        The hazard center is updated using a wind term and a diffusion (random) term.
        The hazard radius is updated as the base radius plus a diffusion-driven spread.
        """
        dt = self.dt
        u = self.wind_u
        v = self.wind_v
        D_diff = self.diffusion_coefficient
        random_dx = np.sqrt(2 * D_diff * dt) * np.random.randn()
        random_dy = np.sqrt(2 * D_diff * dt) * np.random.randn()
        self.hazard_center[0] += u * dt + random_dx
        self.hazard_center[1] += v * dt + random_dy
        self.hazard_center[0] = np.clip(self.hazard_center[0], self.minDimen, self.maxDimen)
        self.hazard_center[1] = np.clip(self.hazard_center[1], self.minDimen, self.maxDimen)
        self.hazard_radius = self.hazard_radius_base + np.sqrt(2 * D_diff * t)
        self.desired_radius = self.hazard_radius + self.hazard_margin

    # ---------- Global Cost Function Calculation ----------
    def _computeGlobalCost(self, p_arr):
        """
        Compute the global cost based on the current hazard parameters and robot positions.
        p_arr is a NumPy array (nr x d).

        Cost components:
          1. Barrier formation cost: sum of squared errors between each robot's distance from the hazard center and the desired barrier radius.
          2. Uniform spacing cost: sum of squared deviations of each angular gap from the ideal gap.
             The ideal gap is defined as: ideal_gap = 2 * desired_radius * sin(pi / noRobots)
        """
        # Barrier formation cost
        dists = np.linalg.norm(p_arr - self.hazard_center, axis=1)
        errors = dists - self.desired_radius
        cost_barrier = np.sum(errors ** 2)

        # Uniform spacing cost: compute angles and gaps
        angles = np.arctan2(p_arr[:, 1] - self.hazard_center[1],
                            p_arr[:, 0] - self.hazard_center[0])
        angles = np.sort(angles)
        gaps = np.diff(np.concatenate([angles, [angles[0] + 2 * np.pi]]))
        ideal_gap = 2 * self.desired_radius * np.sin(np.pi / self.noRobots)
        cost_spacing = np.sum((gaps - ideal_gap) ** 2)

        return cost_barrier + self.lambda_gap * cost_spacing

    def CalculateCF(self, p):
        """
        Calculate the cost function while updating the hazard state.
        """
        self.updateHazard(self.current_iter)
        p_arr = np.array(p, dtype=np.float64)
        return self._computeGlobalCost(p_arr)

    def EvaluateCF(self, p, r):
        """
        Evaluate the cost function without updating the hazard.
        Uses stored hazard parameters.
        """
        p_arr = np.array(p, dtype=np.float64)
        return self._computeGlobalCost(p_arr)

    # ---------- Live Visualization (Matplotlib) ----------
    def initializeLiveVisualization(self):
        """
        Initialize the live visualization.
        In this updated version, we remove the dots and colorbar used previously.
        Instead, we display a background intensity field computed from the hazard intensity.
        """
        if hasattr(self, "_liveVis_initialized") and self._liveVis_initialized:
            return
        plt.ion()
        self._fig = plt.figure(figsize=(8, 6))
        self._ax = self._fig.add_subplot(1, 1, 1)
        # Remove previous colorbar/dot visualization; we'll use imshow for the intensity field.
        self._cost_history = []
        self._trajectories = []
        if self.last_known_decisions is not None:
            p0 = self.last_known_decisions
            for i in range(self.noRobots):
                self._trajectories.append([np.array(p0[i])])
        else:
            for i in range(self.noRobots):
                self._trajectories.append([])
        self._liveVis_initialized = True

    def updateLiveVisualization(self, iteration, cost):
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

        # Compute a grid for the intensity field.
        grid_size = 100
        x = np.linspace(self.minDimen, self.maxDimen, grid_size)
        y = np.linspace(self.minDimen, self.maxDimen, grid_size)
        X, Y = np.meshgrid(x, y)
        XY = np.stack((X, Y), axis=2)
        # Compute hazard intensity at each grid point using a Gaussian model.
        I_field = np.exp(-np.linalg.norm(XY - self.hazard_center, axis=2) ** 2 / (2 * self.hazard_sigma ** 2))
        # Display the intensity field as a background image.
        self._ax.imshow(I_field, extent=[self.minDimen, self.maxDimen, self.minDimen, self.maxDimen],
                        origin='lower', cmap='hot', alpha=0.6)

        # Plot robots
        self._ax.scatter(p_arr[:, 0], p_arr[:, 1], c='blue', s=50, label='Robots')
        # Draw hazard and barrier circles
        hazard_circle = plt.Circle(self.hazard_center, self.hazard_radius, color='red', fill=False, linestyle='--',
                                   label='Hazard')
        barrier_circle = plt.Circle(self.hazard_center, self.desired_radius, color='green', fill=False, linestyle='-',
                                    label='Desired Barrier')
        self._ax.add_artist(hazard_circle)
        self._ax.add_artist(barrier_circle)

        # Draw enclosure: sort robots by angle and connect them
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


# ---------- (Optional) Standalone Test Main ----------
if __name__ == "__main__":
    testbed = HazardEnclosureTestbed()
    if testbed.getLatestDecisionVariables() is None:
        testbed.setInitialDecisionVector(testbed.fetchDecisionVector())
    print("Initial decision vector:", testbed.getLatestDecisionVariables())
    plt.ion()
    for t in range(500):
        p_arr = np.array(testbed.getLatestDecisionVariables(), dtype=np.float64)
        update = np.random.uniform(-0.002, 0.002, p_arr.shape)
        new_positions = p_arr + update
        new_positions = np.clip(new_positions, testbed.minDimen, testbed.maxDimen)
        testbed.setInitialDecisionVector(new_positions.tolist())
        testbed.last_known_decisions = testbed.getLatestDecisionVariables()
        testbed.current_iter = t
        cost = testbed.CalculateCF(testbed.getLatestDecisionVariables())
        testbed.updateLiveVisualization(t, cost)
    testbed.finalizeLiveVisualization()
