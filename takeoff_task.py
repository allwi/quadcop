from task import Task
import numpy as np

class TakeOffTask(Task):

    def __init__(self, target_pos=None, runtime=5., action_repeat=1):
        super().__init__(init_pose=[0.0, 0.0, 5.0, 0.0, 0.0, 0.0], init_velocities=None, init_angle_velocities=None
                         , runtime=runtime, target_pos=target_pos, action_repeat=action_repeat)

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        reward = 0
        if not np.all(self.target_pos == self.init_pos[:3]):
            a = np.cross(self.target_pos - self.sim.pose[:3], self.flight_path)
            reward -= np.linalg.norm(a) / np.linalg.norm(self.flight_path)
        reward -= np.linalg.norm([self.sim.pose[:3] - self.target_pos])

        reward += 10.0/np.abs(self.sim.pose[2] - self.target_pos[2])

        reward = np.power(self.sim.pose[2],2)
        #print('task reward : ',reward)
        return reward

