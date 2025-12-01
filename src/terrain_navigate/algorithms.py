import numpy as np
from heapq import heappush, heappop
from itertools import count
from typing import Tuple, Optional, Dict, Set, List

from .base import PathFinder, Environment, CostFunction

class AStar(PathFinder):
    """
    A* pathfinding algorithm implementation.
    """
    def find_path(
        self, 
        start: np.ndarray, 
        goal: np.ndarray, 
        environment: Environment, 
        cost_function: CostFunction,
        node_radius: float
    ) -> Tuple[Optional[np.ndarray], float]:
        
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        if start.size == 0 or goal.size == 0:
            raise ValueError("Start and goal points cannot be empty.")
        
        if start.shape != goal.shape:
             raise ValueError(f"Start shape {start.shape} and goal shape {goal.shape} must match.")
        
        start_key = tuple(start.tolist())
        goal_key = tuple(goal.tolist())
        
        # Priority queue: (f_score, push_index, node)
        open_heap: List[Tuple[float, int, np.ndarray]] = []
        push_index = count()
        
        # Initial heuristic cost
        h_start = cost_function.heuristic(start, goal)
        heappush(open_heap, (h_start, next(push_index), start.copy()))
        
        came_from: Dict[Tuple[float, float, float], Tuple[float, float, float]] = {}
        g_costs: Dict[Tuple[float, float, float], float] = {start_key: 0.0}
        closed: Set[Tuple[float, float, float]] = set()
        
        while open_heap:
            current_f, _, current = heappop(open_heap)
            current_key = tuple(current.tolist())
            
            if current_key in closed:
                continue
            
            closed.add(current_key)
            
            # Check if we reached the goal (within tolerance or exact match depending on implementation)
            # Here we use exact match on coordinates as keys, but typically we check distance
            if np.allclose(current, goal):
                path_keys = [current_key]
                while path_keys[-1] != start_key:
                    path_keys.append(came_from[path_keys[-1]])
                path_keys.reverse()
                path = np.array(path_keys, dtype=np.float64)
                return path, g_costs[current_key]
            
            neighbors = environment.get_neighbors(current, node_radius)
            
            if neighbors.size == 0:
                continue
                
            for neighbor in neighbors:
                if np.allclose(neighbor, current):
                    continue
                
                neighbor_key = tuple(neighbor.tolist())
                
                # Calculate tentative g score
                cost = cost_function.calculate(current, neighbor)
                tentative_g = g_costs[current_key] + cost
                
                if tentative_g >= g_costs.get(neighbor_key, np.inf):
                    continue
                
                came_from[neighbor_key] = current_key
                g_costs[neighbor_key] = tentative_g
                
                f_score = tentative_g + cost_function.heuristic(neighbor, goal)
                heappush(open_heap, (f_score, next(push_index), neighbor.copy()))
                
        return None, float('inf')
