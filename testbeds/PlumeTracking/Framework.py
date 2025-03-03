#!/usr/bin/env python3
import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Fallback for 2D visualization
import matplotlib.pyplot as plt
from testbeds.testbed_setup import testbed_setup


class PlumeTrackingTestbed(testbed_setup):
    """
    PlumeTrackingTestbed simulates a cooperative multi-UAV plume tracking task using a Gaussian Mixture Model.

    Dynamics:
      - The plume is modeled as a Gaussian mixture. Initially, a single Gaussian component represents the plume.
        After a predefined split time, the plume splits into two (or more) independent components, each evolving
        with its own advection-diffusion dynamics and dispersion (sigma growth).

    Global Objective:
      - A grid over the operational area is used to assign each grid point to its nearest UAV.
      - For each grid point, the local cost is computed as:
              cost(x) = (d_min^2)/(1 + I(x)) - λ_weight * I(x)
        where d_min is the distance to the nearest UAV and I(x) is the combined intensity from all plume components.
      - This formulation prioritizes the negative intensity term when I(x) is high, encouraging the UAVs to
        concentrate around the high-intensity regions of the plume.

    Visualization:
      - A live 2D plot shows the evolving intensity field (from the GMM) along with UAV positions, trajectories,
        and the plume component centers.
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
        self.noRobots = int(self.getParameter("noRobots", 4))  # fewer UAVs increase challenge
        self.d = int(self.getParameter("d", 2))
        self.dt = float(self.getParameter("dt", 0.01))
        self.noIter = int(self.getParameter("noIter", 600))
        self.N = int(self.getParameter("N", 225))  # For background grid visualization

        # Set required attributes from testbed_setup.
        self.nr = self.noRobots
        self.D = self.d

        # Plume (GMM) parameters.
        self.plume_intensity = float(self.getParameter("plume_intensity", 1.0))
        self.diffusion_coefficient = float(self.getParameter("diffusion_coefficient", 0.02))
        self.wind_u = float(self.getParameter("wind_u", 0.1))
        self.wind_v = float(self.getParameter("wind_v", 0.0))
        self.plume_sigma_initial = float(self.getParameter("plume_sigma_initial", 0.5))
        self.plume_sigma_growth = float(self.getParameter("plume_sigma_growth", 0.005))
        self.lambda_weight = float(self.getParameter("lambda_weight", 5.0))  # Increased priority on intensity
        self.split_iter = int(self.getParameter("split_iter", 300))  # iteration when the plume splits

        # Initialize plume components.
        # Each component is a dict with keys: 'center', 'sigma', and 'weight'.
        initial_center = np.array([(self.minDimen + self.maxDimen) / 2.0,
                                   (self.minDimen + self.maxDimen) / 2.0])
        self.plume_components = [{
            'center': initial_center.copy(),
            'sigma': self.plume_sigma_initial,
            'weight': 1.0
        }]

        # Build a base grid for both visualization and cost computation.
        self.worldConstructor()

        # Decision vectors (set externally).
        self.initial_decisions = None
        self.last_known_decisions = None
        self.current_iter = 0  # Updated externally

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
        """Store the decision vector and update last_known_decisions."""
        self.initial_decisions = p
        self.last_known_decisions = p

    def getLatestDecisionVariables(self):
        return self.last_known_decisions

    def isThisAValidDecisionCommand(self, i, p):
        # Check if each component is within [minDimen, maxDimen].
        for j in range(self.d):
            if p[j] < self.minDimen or p[j] > self.maxDimen:
                return False
        return True

    def fetchDecisionVector(self):
        """
        Generate an initial decision vector.
        UAVs are initialized randomly in the lower left quadrant.
        """
        lower_bound = self.minDimen
        upper_bound = (self.minDimen + self.maxDimen) / 2.0
        p = [[float(np.random.uniform(lower_bound, upper_bound)) for _ in range(self.d)]
             for _ in range(self.noRobots)]
        self.setInitialDecisionVector(p)
        return p

    # ---------------- World Construction ----------------
    def worldConstructor(self):
        """
        Construct a uniform grid as the background for visualization and cost computation.
        """
        q = int(np.sqrt(self.N))
        total_points = q * q
        points = np.zeros((total_points, self.d))
        k = 0
        for i in range(1, q + 1):
            for j in range(1, q + 1):
                points[k, 0] = i / (q + 1.0) * self.maxDimen
                points[k, 1] = j / (q + 1.0) * self.maxDimen
                k += 1
        self.base_Q = points
        self.Q = np.copy(self.base_Q)
        self.W = np.ones(total_points)

    # ---------------- Plume Dynamics (Gaussian Mixture Model) ----------------
    def updatePlume(self, t):
        """
        Update each plume component with advection-diffusion dynamics.
        At a specified iteration, if only a single component exists, split it into two.
        """
        dt = self.dt

        # If it's time to split and we only have one component, split it.
        if t >= self.split_iter and len(self.plume_components) == 1:
            comp = self.plume_components[0]
            center = comp['center']
            # Create two new centers with a small offset.
            offset = np.array([0.5, 0.0])
            comp1 = {
                'center': center + offset,
                'sigma': comp['sigma'],
                'weight': 0.5
            }
            comp2 = {
                'center': center - offset,
                'sigma': comp['sigma'],
                'weight': 0.5
            }
            self.plume_components = [comp1, comp2]

        # Update each component.
        for comp in self.plume_components:
            # Update center with wind and diffusion noise.
            random_dx = np.sqrt(2 * self.diffusion_coefficient * dt) * np.random.randn()
            random_dy = np.sqrt(2 * self.diffusion_coefficient * dt) * np.random.randn()
            comp['center'][0] += self.wind_u * dt + random_dx
            comp['center'][1] += self.wind_v * dt + random_dy
            # Clip centers to operational area.
            comp['center'][0] = np.clip(comp['center'][0], self.minDimen, self.maxDimen)
            comp['center'][1] = np.clip(comp['center'][1], self.minDimen, self.maxDimen)
            # Update sigma to simulate dispersion.
            comp['sigma'] += self.plume_sigma_growth * dt

    def _computeGlobalCost(self, p_arr):
        """
        Proposed coverage-based cost function:
          J = sum_over_grid( I(x) * dist(x, nearest UAV) )
        """
        # 1) Compute intensity at each grid point
        M = self.base_Q.shape[0]
        total_intensity = np.zeros(M)
        for comp in self.plume_components:
            diff = self.base_Q - comp['center']
            dists = np.linalg.norm(diff, axis=1)
            # Weighted sum of Gaussians
            total_intensity += (comp['weight'] * self.plume_intensity
                                * np.exp(- (dists ** 2) / (2 * comp['sigma'] ** 2)))

        # 2) Distance from each grid point to the nearest UAV
        diff_uav = self.base_Q[:, np.newaxis, :] - p_arr[np.newaxis, :, :]  # shape (M, n, 2)
        distances = np.linalg.norm(diff_uav, axis=2)  # shape (M, n)
        d_min = np.min(distances, axis=1)  # shape (M,)

        # 3) Weighted coverage cost
        # The cost is the sum of I(x) * d_min(x)
        coverage_cost = total_intensity * d_min

        return np.sum(coverage_cost)

    # ---------------- Global Cost Function Methods ----------------
    def CalculateCF(self, p):
        """
        Calculate the global cost function while updating the plume state.
        """
        self.updatePlume(self.current_iter)
        p_arr = np.array(p, dtype=np.float64)
        return self._computeGlobalCost(p_arr)

    def EvaluateCF(self, p, r):
        """
        Evaluate the cost function without updating the plume state.
        (r is ignored as the cost is global.)
        """
        p_arr = np.array(p, dtype=np.float64)
        return self._computeGlobalCost(p_arr)

    # ---------------- Live Visualization Methods ----------------
    def initializeLiveVisualization(self):
        """
        Initialize live visualization.
        Displays the evolving GMM plume intensity field, UAV positions, trajectories, and plume centers.
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
        Update live visualization: plot the evolving GMM plume intensity field, UAV trajectories, and plume centers.
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

        # Create a grid for the intensity field.
        grid_size = 100
        x = np.linspace(self.minDimen, self.maxDimen, grid_size)
        y = np.linspace(self.minDimen, self.maxDimen, grid_size)
        X, Y = np.meshgrid(x, y)
        XY = np.stack((X, Y), axis=2)
        # Compute the combined intensity from all plume components over the grid.
        I_field = np.zeros(XY.shape[:2])
        for comp in self.plume_components:
            dist_field = np.linalg.norm(XY - comp['center'], axis=2)
            I_field += comp['weight'] * self.plume_intensity * np.exp(- (dist_field ** 2) / (2 * comp['sigma'] ** 2))
        self._ax.imshow(I_field, extent=[self.minDimen, self.maxDimen, self.minDimen, self.maxDimen],
                        origin='lower', cmap='viridis', alpha=0.6)

        # Plot UAV positions.
        self._ax.scatter(p_arr[:, 0], p_arr[:, 1], c='blue', s=50, label='UAVs')
        # Plot plume component centers.
        for comp in self.plume_components:
            self._ax.scatter(comp['center'][0], comp['center'][1], c='red', s=80, marker='*', label='Plume Center')
        # Draw trajectories for each UAV.
        for i in range(self.noRobots):
            traj_arr = np.array(self._trajectories[i])
            self._ax.plot(traj_arr[:, 0], traj_arr[:, 1], 'c-', linewidth=1)

        self._ax.set_title(f"Iteration {iteration} | CF: {cost:.3f}")
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

    # ---------------- Additional (Optional) Methods ----------------
    def updateAugmentedDecisionVector(self, A):
        """
        Overridden helper to update the latest decisions.
        """
        import copy
        self.last_known_decisions = copy.deepcopy(A)
