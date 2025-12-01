from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional

class Environment(ABC):
    """
    Abstract base class for the environment/map representation.
    """
    @abstractmethod
    def get_neighbors(self, node: np.ndarray, radius: float) -> np.ndarray:
        """
        Finds neighbor nodes within a given radius.
        
        Args:
            node: The current node coordinates (x, y, z).
            radius: The search radius.
            
        Returns:
            An array of neighbor node coordinates.
        """
        pass

    @abstractmethod
    def get_nearest_node(self, point: np.ndarray, radius: Optional[float] = None) -> np.ndarray:
        """
        Finds the nearest node in the environment to a given point.
        
        Args:
            point: The query point coordinates.
            radius: The search radius. If None, a default radius based on environment scale is used.
            
        Returns:
            The coordinates of the nearest node.
        """
        pass

class CostFunction(ABC):
    """
    Abstract base class for cost calculation strategies.
    """
    @abstractmethod
    def calculate(self, node_a: np.ndarray, node_b: np.ndarray) -> float:
        """
        Calculates the cost to move from node_a to node_b.
        """
        pass

    @abstractmethod
    def heuristic(self, node: np.ndarray, goal: np.ndarray) -> float:
        """
        Calculates the heuristic cost from node to goal.
        """
        pass

class PathFinder(ABC):
    """
    Abstract base class for pathfinding algorithms.
    """
    @abstractmethod
    def find_path(
        self, 
        start: np.ndarray, 
        goal: np.ndarray, 
        environment: Environment, 
        cost_function: CostFunction,
        node_radius: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Finds a path from start to goal.
        
        Args:
            start: Start node coordinates.
            goal: Goal node coordinates.
            environment: The environment to search in.
            cost_function: The cost function to use.
            node_radius: Radius for finding neighbors.
            
        Returns:
            A tuple containing the path (as an array of points) and the total cost.
            Returns (None, infinity) if no path is found.
        """
        pass
