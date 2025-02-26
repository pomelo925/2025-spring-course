import cv2
import sys
import heapq
sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner

class PlannerAStar(Planner):
    def __init__(self, m, inter=10):
        super().__init__(m)
        self.inter = inter
        self.initialize()

    def initialize(self):
        self.queue = []
        self.parent = {}
        self.h = {} # Distance from start to node
        self.g = {} # Distance from node to goal
        self.goal_node = None

    def planning(self, start=(100,200), goal=(375,520), inter=None, img=None):
        if inter is None:
            inter = self.inter
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        # Initialize 
        self.initialize()
        heapq.heappush(self.queue, (0, start))
        self.parent[start] = None
        self.g[start] = 0
        self.h[start] = utils.distance(start, goal)
        
        while self.queue:
            current = heapq.heappop(self.queue)
            
            # If the goal is reached, set the goal node and break
            if current == goal:
                self.goal_node = current
                break
            
            # Get neighbors of the current node
            neighbors = utils.get_neighbors(current, self.m)

            for neighbor in neighbors:
                # Calculate tentative g score
                tentative_g = self.g[current] + utils.distance(current, neighbor)
            
                # If this path to neighbor is better, update the path
                if neighbor not in self.g or tentative_g < self.g[neighbor]:
                    self.g[neighbor] = tentative_g
                    f = tentative_g + utils.distance(neighbor, goal)
                    heapq.heappush(self.queue, (f, neighbor))
                    self.parent[neighbor] = current
                
        # Extract path
        path = []
        p = self.goal_node
        if p is None:
            return path
        while True:
            path.insert(0, p)
            if self.parent[p] is None:
                break
            p = self.parent[p]
        if path[-1] != goal:
            path.append(goal)
        return path