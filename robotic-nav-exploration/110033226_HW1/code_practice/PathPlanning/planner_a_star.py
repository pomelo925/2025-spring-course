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
        self.open_list = []
        self.closed_list = set()
        self.parent = {}
        self.g = {}
        self.h = {}
        self.goal_node = None

    def planning(self, start=(100, 200), goal=(375, 520), inter=None, img=None):
        if inter is None:
            inter = self.inter
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        self.initialize()
        self.g[start] = 0
        self.h[start] = utils.distance(start, goal)
        heapq.heappush(self.open_list, (self.g[start] + self.h[start], start))

        while self.open_list:
            # Get the node with the smallest value
            _, current_node = heapq.heappop(self.open_list)

            # Skip if visited
            if current_node in self.closed_list:
                continue

            # Stop if reached
            if utils.distance(current_node, goal) < inter:
                self.goal_node = current_node
                break

            # Mark the node as visited
            self.closed_list.add(current_node)

            # Calculate neighboring nodes
            neighbors = [
                (current_node[0] + inter, current_node[1]),
                (current_node[0], current_node[1] + inter),
                (current_node[0] - inter, current_node[1]),
                (current_node[0], current_node[1] - inter),
                (current_node[0] + inter, current_node[1] + inter),
                (current_node[0] - inter, current_node[1] + inter),
                (current_node[0] - inter, current_node[1] - inter),
                (current_node[0] + inter, current_node[1] - inter)
            ]

            # Ignore invalid nodes: obstacles or visited 
            for neighbor in neighbors:
                if self.map[neighbor[1], neighbor[0]] < 0.5 or neighbor in self.closed_list:
                    continue

                tentative_g = self.g[current_node] + inter
                if neighbor not in self.g or tentative_g < self.g[neighbor]:
                    # update path: first visit or shorter path found
                    self.g[neighbor] = tentative_g
                    self.h[neighbor] = utils.distance(neighbor, goal)
                    self.parent[neighbor] = current_node

                    heapq.heappush(self.open_list, (self.g[neighbor] + self.h[neighbor], neighbor))

        path = []
        p = self.goal_node
        while p is not None:
            path.insert(0, p)
            p = self.parent.get(p)
        return path