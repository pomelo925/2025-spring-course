import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerLQRBicycle(Controller):
    def __init__(self, Q=np.eye(4), R=np.eye(1)):
        self.path = None
        self.Q = Q
        self.R = R
        self.pe = 0
        self.pth_e = 0

    def set_path(self, path):
        super().set_path(path)
        self.pe = 0
        self.pth_e = 0

    def _solve_DARE(self, A, B, Q, R, max_iter=150, eps=1e-6):
        P = Q.copy()
        for _ in range(max_iter):
            P_next = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
            if np.max(np.abs(P_next - P)) < eps:
                break
            P = P_next
        return P_next

    def feedback(self, info):
        if self.path is None:
            print("No path !!")
            return None, None

        x, y, theta, v, L, dt = info["x"], info["y"], info["yaw"], info["v"], info["l"], info["dt"]
        theta = utils.angle_norm(theta)
        
        min_idx, _ = utils.search_nearest(self.path, (x, y))
        target = self.path[min_idx]
        target_theta = utils.angle_norm(target[2])
        
        e = np.hypot(target[0] - x, target[1] - y)
        e_dot = (e - self.pe) / dt
        theta_e = utils.angle_norm(target_theta - theta)
        theta_e_dot = (theta_e - self.pth_e) / dt
        self.pe, self.pth_e = e, theta_e
        
        A = np.array([[1, dt, 0, 0],
                      [0, 0, v, 0],
                      [0, 0, 1, dt],
                      [0, 0, 0, 0]])
        B = np.array([[0],
                      [0],
                      [0],
                      [v / L]])
        
        P = self._solve_DARE(A, B, self.Q, self.R)
        K = np.linalg.inv(self.R + B.T @ P @ B) @ B.T @ P @ A
        
        state = np.array([[e], [e_dot], [theta_e], [theta_e_dot]])
        delta = (K @ state).item()
        
        return delta