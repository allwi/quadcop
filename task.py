import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]), init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None, action_repeat=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = action_repeat if action_repeat is not None else 3

        self.init_pos = init_pose
        self.state_size = self.action_repeat * 6
        self.action_low = 10
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.flight_path = self.target_pos - self.init_pos[:3]
        self.num_steps = 0
        #np.seterr(all='raise')

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        reward = 0
        if not np.all(self.target_pos == self.init_pos[:3]):
            a = np.cross(self.target_pos - self.sim.pose[:3], self.flight_path)
            reward -= np.linalg.norm(a) / np.linalg.norm(self.flight_path)
        reward -= np.linalg.norm([self.sim.pose[:3] - self.target_pos])

        # print('task reward : ',reward)
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        self.num_steps += 1
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.num_steps = 0
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state