import numpy as np
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
