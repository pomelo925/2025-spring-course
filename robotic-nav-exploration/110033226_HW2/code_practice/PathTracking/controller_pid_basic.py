import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPIDBasic(Controller):
    def __init__(self, kp_v=0.2, ki_v=0.0, kd_v=0.0, 
                 kp_w=1500, ki_w=0.0, kd_w=0.):
        self.path = None
        self.pid_v = np.array([kp_v, ki_v, kd_v])
        self.pid_w = np.array([kp_w, ki_w, kd_w])
        self.acc_ev = 0
        self.last_ev = 0
        self.acc_ew = 0
        self.last_ew = 0
        self.goal_threshold = 100
        self.temp_index = 0  # Initialize temp_index
    
    def set_path(self, path):
        super().set_path(path)
        self.acc_ev = 0
        self.last_ev = 0
        self.acc_ew = 0
        self.last_ew = 0
        self.temp_index = 0  # Reset temp_index
    
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State
        x, y, dt = info["x"], info["y"], info["dt"]
        v = info["v"]

        # Search Nearest Target
        min_idx, min_dist = utils.search_nearest(self.path, (x,y))
        
        # Ensure index never becomes smaller
        if min_idx >= self.temp_index:
            self.temp_index = min_idx
        target = self.path[self.temp_index]
        
        # Check if close to the current target
        if min_dist < self.goal_threshold:
            if self.temp_index < len(self.path) - 1:
                target = self.path[self.temp_index + 1]
            else:
                print("Goal reached!")
                return 0, 0
        
        ## PID Control for angular velocity
        # error
        error_w = np.arctan2(target[1] - y, target[0] - x) - info["yaw"]
        error_w = 2*3.14159*np.arctan2(np.sin(error_w), np.cos(error_w))  # Normalize the angle error
        
        # P gain
        p_term_w = self.pid_w[0] * error_w
        
        # I gain
        self.acc_ew += error_w * dt
        i_term_w = self.pid_w[1] * self.acc_ew
        
        # D gain
        d_term_w = self.pid_w[2] * (error_w - self.last_ew) / dt
        self.last_ew = error_w
        
        next_w = p_term_w + i_term_w + d_term_w
        
        ## PID Control for linear velocity
        # error
        error_v = np.hypot(target[0] - x, target[1] - y)
        
        # P gain
        p_term_v = self.pid_v[0] * error_v
        
        # I gain
        self.acc_ev += error_v * dt
        i_term_v = self.pid_v[1] * self.acc_ev
        
        # D gain
        d_term_v = self.pid_v[2] * (error_v - self.last_ev) / dt
        self.last_ev = error_v
        
        next_v = p_term_v + i_term_v + d_term_v
        
        # debug message
        self.pid_print(target, self.temp_index, min_dist, error_v, p_term_v, i_term_v, d_term_v, error_w, p_term_w, i_term_w, d_term_w)
        
        return next_v, next_w
    
    def pid_print(self, target, temp_index, min_dist, error_v, p_term_v, i_term_v, d_term_v, error_w, p_term_w, i_term_w, d_term_w):
        # Erase all output
        sys.stdout.write("\033[2J\033[H")

        print(f"\033[94m[Target]\tpos: ({target[0]:.2f}, {target[1]:.2f})\tIndex: {temp_index}\tDist: {min_dist:.2f}\033[0m")

        print(f"\033[93m[Linear]\terr: {error_v:.2f}\tP: {p_term_v:.2f}\tI={i_term_v:.2f}\tD={d_term_v:.2f}\033[0m")
        print(f"\033[93m[Angular]\terr: {error_w:.2f}\tp: {p_term_w:.2f}\tI={i_term_w:.2f}\tD={d_term_w:.2f}\033[0m")
