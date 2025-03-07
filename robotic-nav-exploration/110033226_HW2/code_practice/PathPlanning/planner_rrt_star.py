import cv2
import numpy as np
import sys
sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner

class PlannerRRTStar(Planner):
    def __init__(self, m, extend_len=20):
        super().__init__(m)
        self.extend_len = extend_len 

    def _random_node(self, goal, shape):
        r = np.random.choice(2,1,p=[0.5,0.5])
        if r==1:
            return (float(goal[0]), float(goal[1]))
        else:
            rx = float(np.random.randint(int(shape[1])))
            ry = float(np.random.randint(int(shape[0])))
            return (rx, ry)

    def _nearest_node(self, samp_node):
        min_dist = 99999
        min_node = None
        for n in self.ntree:
            dist = utils.distance(n, samp_node)
            if dist < min_dist:
                min_dist = dist
                min_node = n
        return min_node

    def _check_collision(self, n1, n2):
        n1_ = utils.pos_int(n1)
        n2_ = utils.pos_int(n2)
        line = utils.Bresenham(n1_[0], n2_[0], n1_[1], n2_[1])
        for pts in line:
            if self.map[int(pts[1]),int(pts[0])]<0.5:
                return True
        return False

    def _steer(self, from_node, to_node, extend_len):
        vect = np.array(to_node) - np.array(from_node)
        v_len = np.hypot(vect[0], vect[1])
        v_theta = np.arctan2(vect[1], vect[0])
        if extend_len > v_len:
            extend_len = v_len
        new_node = (from_node[0]+extend_len*np.cos(v_theta), from_node[1]+extend_len*np.sin(v_theta))
        if new_node[1]<0 or new_node[1]>=self.map.shape[0] or new_node[0]<0 or new_node[0]>=self.map.shape[1] or self._check_collision(from_node, new_node):
            return False, None
        else:        
            return new_node, utils.distance(new_node, from_node)
    

    def planning(self, start, goal, extend_len=None, img=None):
        if extend_len is None:
            extend_len = self.extend_len
        self.ntree = {}
        self.ntree[start] = None
        self.cost = {}
        self.cost[start] = 0
        goal_node = None
        
        r_near = extend_len * 2  # Radius for re-parenting and re-wiring

        for it in range(20000):
            samp_node = self._random_node(goal, self.map.shape)
            near_node = self._nearest_node(samp_node)
            new_node, cost = self._steer(near_node, samp_node, extend_len)

            if new_node is False:
                continue
            
            # Find nearest neighbors
            near_neighbors = [
                n for n in self.ntree if utils.distance(n, new_node) < r_near
            ]
            
            # Re-parenting (choose best parent)
            best_parent = near_node
            min_cost = self.cost[near_node] + cost
            for neighbor in near_neighbors:
                new_cost = self.cost[neighbor] + utils.distance(neighbor, new_node)
                if new_cost < min_cost and not self._check_collision(neighbor, new_node):
                    best_parent = neighbor
                    min_cost = new_cost
            
            # Add new node with best parent
            self.ntree[new_node] = best_parent
            self.cost[new_node] = min_cost

            # Rewire neighbors to potentially reduce their cost
            for neighbor in near_neighbors:
                new_cost = self.cost[new_node] + utils.distance(new_node, neighbor)
                if new_cost < self.cost[neighbor] and not self._check_collision(new_node, neighbor):
                    self.ntree[neighbor] = new_node
                    self.cost[neighbor] = new_cost

            if utils.distance(new_node, goal) < extend_len:
                goal_node = new_node
                break

            # Visualization
            if img is not None:
                for n in self.ntree:
                    if self.ntree[n] is None:
                        continue
                    node = self.ntree[n]
                    cv2.line(img, (int(n[0]), int(n[1])), (int(node[0]), int(node[1])), (0,1,0), 1)
                img_ = img.copy()
                cv2.circle(img_, utils.pos_int(new_node), 5, (0,0.5,1), 3)
                img_ = cv2.flip(img_, 0)
                cv2.imshow("Path Planning", img_)
                k = cv2.waitKey(1)
                if k == 27:
                    break
        
        # Extract Path
        path = []
        n = goal_node
        while n is not None:
            path.insert(0, n)
            n = self.ntree[n]
        path.append(goal)

        return path