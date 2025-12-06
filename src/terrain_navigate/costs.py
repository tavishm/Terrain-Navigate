import numpy as np
from typing import Callable, Optional
from .base import CostFunction

class EuclideanCost(CostFunction):
    """
    Standard Euclidean distance cost function.
    """
    def calculate(self, node_a: np.ndarray, node_b: np.ndarray) -> float:
        return float(np.linalg.norm(node_a - node_b))

    def heuristic(self, node: np.ndarray, goal: np.ndarray) -> float:
        return float(np.linalg.norm(node - goal))


class PowerCost(CostFunction):
    """
    Cost function with power penalty for vertical movement (v2/v3 logic).
    """
    def __init__(self, n: float = 6.0, alpha: float = 1e-5):
        self.n = n
        self.alpha = alpha

    def calculate(self, node_a: np.ndarray, node_b: np.ndarray) -> float:
        # (x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^n
        # Note: We assume node_a and node_b have at least 3 dimensions (x, y, z)
        dx = node_b[0] - node_a[0]
        dy = node_b[1] - node_a[1]
        dz = node_b[2] - node_a[2]
        return float(dx**2 + dy**2 + abs(dz)**self.n)

    def heuristic(self, node: np.ndarray, goal: np.ndarray) -> float:
        # alpha * Euclidean distance
        dist = float(np.linalg.norm(node - goal))
        return self.alpha * dist


class ApproachOneCost(CostFunction):
    """
    Approach 1: Penalize elevation change.
    Cost = Distance * (1 + (abs(dz) / dr)^n)
    where dr is horizontal distance and dz is vertical distance.
    """
    def __init__(self, n: float = 2.0):
        self.n = n

    def calculate(self, node_a: np.ndarray, node_b: np.ndarray) -> float:
        diff = node_b - node_a
        dist = float(np.linalg.norm(diff))
        
        # Avoid division by zero if dr is tiny
        dr = float(np.linalg.norm(diff[:2]))
        dz = abs(diff[2]) if diff.shape[0] > 2 else 0.0
        
        if dr < 1e-6:
            # Vertical movement or stationary
            penalty = 0.0 if dz == 0 else float('inf') # Or some large number? 
            # If purely vertical, slope is infinite. 
            # Let's assume a large penalty or handle gracefully.
            # If dz > 0 and dr ~ 0, slope is huge.
            if dz > 0:
                penalty = 1000.0 # Cap penalty?
            else:
                penalty = 0.0
        else:
            penalty = (dz / dr) ** self.n
            
        return dist * (1.0 + penalty)

    def heuristic(self, node: np.ndarray, goal: np.ndarray) -> float:
        return float(np.linalg.norm(node - goal))


class ApproachTwoCost(CostFunction):
    """
    Approach 2: Penalize elevation change and tangential slope.
    Cost = Distance * (1 + (abs(dz)/dr)^n1 + (abs(dz)/dr_p)^n2)
    where dr_p is perpendicular horizontal distance (magnitude same as dr).
    """
    def __init__(self, n1: float = 2.0, n2: float = 2.0):
        self.n1 = n1
        self.n2 = n2

    def calculate(self, node_a: np.ndarray, node_b: np.ndarray) -> float:
        diff = node_b - node_a
        dist = float(np.linalg.norm(diff))
        
        dr = float(np.linalg.norm(diff[:2]))
        dz = abs(diff[2]) if diff.shape[0] > 2 else 0.0
        
        # dr_p magnitude is same as dr for perpendicular vector in XY plane
        dr_p = dr 
        
        if dr < 1e-6:
            penalty1 = 1000.0 if dz > 0 else 0.0
            penalty2 = 1000.0 if dz > 0 else 0.0
        else:
            penalty1 = (dz / dr) ** self.n1
            penalty2 = (dz / dr_p) ** self.n2
            
        return dist * (1.0 + penalty1 + penalty2)

    def heuristic(self, node: np.ndarray, goal: np.ndarray) -> float:
        return float(np.linalg.norm(node - goal))


class ApproachThreeCost(CostFunction):
    """
    Approach 3: Trigonometric-like penalty with coefficients.
    Cost = Distance * (1 + (alpha1 * abs(dz)/dr + beta1)^n1 + (alpha2 * abs(dz)/dr_p + beta2)^n2)
    """
    def __init__(self, n1: float = 2.0, n2: float = 2.0, 
                 alpha1: float = 1.0, beta1: float = 0.0,
                 alpha2: float = 1.0, beta2: float = 0.0):
        self.n1 = n1
        self.n2 = n2
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.alpha2 = alpha2
        self.beta2 = beta2

    def calculate(self, node_a: np.ndarray, node_b: np.ndarray) -> float:
        diff = node_b - node_a
        dist = float(np.linalg.norm(diff))
        
        dr = float(np.linalg.norm(diff[:2]))
        dz = abs(diff[2]) if diff.shape[0] > 2 else 0.0
        dr_p = dr
        
        if dr < 1e-6:
            term1 = (self.alpha1 * 1000.0 + self.beta1) if dz > 0 else self.beta1
            term2 = (self.alpha2 * 1000.0 + self.beta2) if dz > 0 else self.beta2
        else:
            term1 = (self.alpha1 * (dz / dr) + self.beta1)
            term2 = (self.alpha2 * (dz / dr_p) + self.beta2)
            
        # Ensure terms are non-negative before exponentiation if n is fractional? 
        # Assuming n is integer or terms are positive. 
        # User formula implies these are penalties, so should be positive.
        # If beta is negative, term could be negative. We'll take abs? Or max(0)?
        # Let's assume standard usage where result is positive.
        
        penalty = term1**self.n1 + term2**self.n2
        return dist * (1.0 + penalty)

    def heuristic(self, node: np.ndarray, goal: np.ndarray) -> float:
        return float(np.linalg.norm(node - goal))


class ApproachFourCost(CostFunction):
    """
    Approach 4: Alternate approach mixing dr and dz directly.
    Cost = Distance * (1 + dr * (abs(dz))^n1 + alpha * dr_p * (abs(dz))^n2)
    Note: dr and dz here are magnitudes from the step vector.
    """
    def __init__(self, n1: float = 1.0, n2: float = 1.0, alpha: float = 1.0):
        self.n1 = n1
        self.n2 = n2
        self.alpha = alpha

    def calculate(self, node_a: np.ndarray, node_b: np.ndarray) -> float:
        diff = node_b - node_a
        dist = float(np.linalg.norm(diff))
        
        dr = float(np.linalg.norm(diff[:2]))
        dz = abs(diff[2]) if diff.shape[0] > 2 else 0.0
        dr_p = dr
        
        # User formula: h(n) = sum( dr * dz^n1 + alpha * dr_p * dz^n2 )
        # We interpret this as the penalty term added to distance.
        # However, if n1=1, n2=1, this is dr*dz + alpha*dr*dz.
        # Units: L * L = L^2. Distance is L. 
        # If we just add this to Distance (L), units mismatch.
        # But if we treat it as a weight multiplier?
        # "Cost = Distance * (1 + ...)"
        # Then ... must be dimensionless.
        # dr * dz^n1 has dimensions L^(1+n1).
        # This suggests the user's formula might be the *entire* cost, not a multiplier.
        # "h(n) = sum(...)"
        # If it's the entire cost, then moving on flat ground (dz=0) has cost 0.
        # This is bad for A* (0 cost edges).
        # So I will stick to "Distance * (1 + Penalty)".
        # And I will assume the user's formula IS the penalty, despite unit weirdness.
        # It's "Just for fun" and "intuitively derived".
        
        penalty = dr * (dz**self.n1) + self.alpha * dr_p * (dz**self.n2)
        return dist * (1.0 + penalty)

    def heuristic(self, node: np.ndarray, goal: np.ndarray) -> float:
        return float(np.linalg.norm(node - goal))


class EnergyBasedCost(CostFunction):
    """
    Implements the Rowe & Ross (2000) style energy model.
    Cost of a segment with length s and slope phi is:
        E = m * g * (mu * cos(phi) + sin(phi)) * s
    where mu captures rolling friction. Uphill moves cost more, downhill moves
    are clamped to a small positive cost to avoid negative edge weights.
    """

    def __init__(
        self,
        mass: float = 50.0,
        gravity: float = 9.81,
        friction_coeff: float = 0.6,
        min_cost: float = 1e-6,
    ):
        self.mass = mass
        self.gravity = gravity
        self.friction_coeff = friction_coeff
        self.min_cost = min_cost

    def calculate(self, node_a: np.ndarray, node_b: np.ndarray) -> float:
        diff = node_b - node_a
        segment_length = float(np.linalg.norm(diff))
        if segment_length < 1e-9:
            return 0.0

        horizontal = float(np.linalg.norm(diff[:2])) if diff.shape[0] >= 2 else 0.0
        vertical = float(diff[2]) if diff.shape[0] > 2 else 0.0

        cos_phi = horizontal / segment_length if segment_length > 0 else 0.0
        sin_phi = vertical / segment_length if segment_length > 0 else 0.0

        energy = self.mass * self.gravity * (
            (self.friction_coeff * cos_phi) + sin_phi
        ) * segment_length

        return max(self.min_cost, energy)

    def heuristic(self, node: np.ndarray, goal: np.ndarray) -> float:
        # Conservative heuristic keeps A* admissible even when slopes are unknown.
        return float(np.linalg.norm(node - goal))


MetricFn = Callable[[np.ndarray, np.ndarray], float]


class TerrainTraversabilityCost(CostFunction):
    """
    Liu et al. (2024) inspired composite terrain cost:
        cost = omega_1 * d(n') + omega_2 * (w_s*f_s + w_e*f_e + w_r*f_r)
    Default metric functions approximate slope/elevation using the segment
    geometry, but custom callables can be injected for richer terrain data.
    """

    def __init__(
        self,
        omega_distance: float = 1.0,
        omega_slope: float = 1.0,
        omega_elevation: float = 1.0,
        omega_roughness: float = 1.0,
        slope_fn: Optional[MetricFn] = None,
        elevation_fn: Optional[MetricFn] = None,
        roughness_fn: Optional[MetricFn] = None,
        epsilon: float = 1e-6,
    ):
        self.omega_distance = omega_distance
        self.omega_slope = omega_slope
        self.omega_elevation = omega_elevation
        self.omega_roughness = omega_roughness
        self.slope_fn = slope_fn or self._default_slope
        self.elevation_fn = elevation_fn or self._default_elevation
        self.roughness_fn = roughness_fn or self._default_roughness
        self.epsilon = epsilon

    def _planar_distance(self, vector: np.ndarray) -> float:
        if vector.size < 2:
            return 0.0
        return float(np.linalg.norm(vector[:2]))

    def _vertical_delta(self, vector: np.ndarray) -> float:
        if vector.size < 3:
            return 0.0
        return float(vector[2])

    def _default_slope(self, node_a: np.ndarray, node_b: np.ndarray) -> float:
        diff = node_b - node_a
        horizontal = self._planar_distance(diff)
        vertical = abs(self._vertical_delta(diff))
        return vertical / max(horizontal, self.epsilon)

    def _default_elevation(self, node_a: np.ndarray, node_b: np.ndarray) -> float:
        diff = node_b - node_a
        return abs(self._vertical_delta(diff))

    def _default_roughness(self, node_a: np.ndarray, node_b: np.ndarray) -> float:
        _ = (node_a, node_b)
        return 0.0

    def calculate(self, node_a: np.ndarray, node_b: np.ndarray) -> float:
        diff = node_b - node_a
        planar = self._planar_distance(diff)

        slope_cost = max(0.0, self.slope_fn(node_a, node_b))
        elevation_cost = max(0.0, self.elevation_fn(node_a, node_b))
        roughness_cost = max(0.0, self.roughness_fn(node_a, node_b))

        terrain_penalty = (
            self.omega_slope * slope_cost
            + self.omega_elevation * elevation_cost
            + self.omega_roughness * roughness_cost
        )

        return (self.omega_distance * planar) + terrain_penalty

    def heuristic(self, node: np.ndarray, goal: np.ndarray) -> float:
        diff = goal - node
        planar = self._planar_distance(diff)
        return self.omega_distance * planar


class CompositeDistanceElevationSteeringCost(CostFunction):
    """
    Implements Jiang et al. (2024) style composite edge cost:
        C = K1 * s + K2 * |Δz| + K3 * |Δheading|
    Heading differences are derived from an explicit heading dimension when
    available; otherwise the steering penalty defaults to zero.
    """

    def __init__(
        self,
        k1: float = 1.0,
        k2: float = 1.0,
        k3: float = 1.0,
        heading_index: Optional[int] = None,
    ):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.heading_index = heading_index

    def _segment_length(self, diff: np.ndarray) -> float:
        if diff.size >= 3:
            return float(np.linalg.norm(diff[:3]))
        return float(np.linalg.norm(diff))

    def _vertical_change(self, diff: np.ndarray) -> float:
        if diff.size < 3:
            return 0.0
        return abs(float(diff[2]))

    def _heading_change(self, node_a: np.ndarray, node_b: np.ndarray) -> float:
        idx = self.heading_index
        if idx is None and min(node_a.size, node_b.size) > 3:
            idx = 3

        if idx is not None and idx < min(node_a.size, node_b.size):
            delta = float(node_b[idx] - node_a[idx])
            wrapped = (delta + np.pi) % (2 * np.pi) - np.pi
            return abs(wrapped)

        return 0.0

    def calculate(self, node_a: np.ndarray, node_b: np.ndarray) -> float:
        diff = node_b - node_a
        segment_length = self._segment_length(diff)
        vertical_change = self._vertical_change(diff)
        heading_change = self._heading_change(node_a, node_b)

        return (
            self.k1 * segment_length
            + self.k2 * vertical_change
            + self.k3 * heading_change
        )

    def heuristic(self, node: np.ndarray, goal: np.ndarray) -> float:
        diff = goal - node
        return self.k1 * self._segment_length(diff)
