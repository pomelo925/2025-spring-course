
import cv2
import sys
import heapq
import numpy as np
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
        self.h = {}  # Distance from start to node
        self.g = {}  # Distance from node to goal
        self.goal_node = None

    def planning(self, start=(100, 200), goal=(375, 520), inter=None, img=None):
        if inter is None:
            inter = self.inter
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        # Initialize the algorithm
        self.initialize()
        self.queue.append((0, start))  # `pri`ority queue with (cost, node)
        self.parent[start] = None
        self.g[start] = 0
        self.h[start] = utils.distance(start, goal)
        
        img_copy = img.copy()
        
        while self.queue:
            # Get the node with the lowest cost
            _, current = heapq.heappop(self.queue)

            # Visualization:　current node
            cv2.circle(img_copy, current, 2, (255, 0, 0), -1)
            cv2.imshow("A* Pathfinding", img_copy)

            if cv2.waitKey(1) == 27:
                break
            
            # Check if the goal is reached
            if current == goal:
                self.goal_node = current
                break
            
            # Explore the neighbors
            for i in range(-1, 2):
                for j in range(-1, 2):
                    # Skip current node
                    if i == 0 and j == 0:
                        continue
                    # Skip diagonal nodes
                    new_node = (current[0] + i * inter, current[1] + j * inter)
                    
                    # Check if the node is valid
                    if (new_node[1] < 0 or new_node[1] >= self.map.shape[0] or 
                        new_node[0] < 0 or new_node[0] >= self.map.shape[1] or 
                        self.map[new_node[1], new_node[0]] < 0.5):
                        continue
                    
                    # Calculate the cost
                    new_cost = self.g[current] + utils.distance(current, new_node)

                    # Update the cost if it is lower
                    if new_node not in self.g or new_cost < self.g[new_node]:
                        self.parent[new_node] = current
                        self.g[new_node] = new_cost
                        self.h[new_node] = utils.distance(new_node, goal)
                        heapq.heappush(self.queue, (self.g[new_node] + self.h[new_node], new_node))
        
        # Extract path
        path = []
        p = self.goal_node
        if p is None:
            return path
        
        while p is not None:
            path.insert(0, p)
            p = self.parent[p]
        
        # Visualization
        for i in range(len(path) - 1):
            cv2.line(img_copy, path[i], path[i + 1], (0, 255, 0), 2)
        
        cv2.imshow("A* Pathfinding", img_copy)
        
        # Wait for ESC key to close
        while True:
            if cv2.waitKey(0) == 27:
                break
        cv2.destroyAllWindows()
        return path