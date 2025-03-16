import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPurePursuitBasic(Controller):
    def __init__(self, kp=1, Lfc=10):
        self.path = None
        self.kp = kp
        self.Lfc = Lfc

    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, v = info["x"], info["y"], info["yaw"], info["v"]

        # Search Front Target
        min_idx, _ = utils.search_nearest(self.path, (x, y))
        Ld = self.kp * v + self.Lfc

        target_idx = next((i for i in range(min_idx, len(self.path)-1) 
                    if np.sqrt((self.path[i+1,0]-x)**2 + (self.path[i+1,1]-y)**2) > Ld), len(self.path)-1)

        # Pure Pursuit Control for Basic Kinematic Model
        target_angle = np.arctan2(self.path[target_idx, 1] - y, self.path[target_idx, 0] - x) - np.deg2rad(yaw)
        next_w = np.rad2deg(2 * v * np.sin(target_angle) / Ld)

        return next_w