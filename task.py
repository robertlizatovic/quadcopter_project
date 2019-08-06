import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
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
        self.action_repeat = 3
        state_s = len(self.get_state())

        self.state_size = self.action_repeat * state_s
        self.action_low = 850
        self.action_high = 860
        self.action_size = 4

        # Goal: reach a target coordinate (x, y, z) as fast as possible and hover there for the
        # remainder of the episode.
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_state(self):
        """returns the state observable by the agent from the simulation object"""
        # state contains the following: x, y, z, phi, theta, psi, Vx, Vy, Vz, wx, wy, wz
        return np.concatenate((self.sim.pose, self.sim.v, self.sim.angular_v))

    def get_reward(self, done):
        """Uses current pose of sim to return reward."""
        # distance reward - how far is the drone from the target position
        distance = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        # distance normalized to a (0, +1] range
        z = self.sim.pose[2]
        z_target = self.target_pos[2]
        if z < z_target:
            reward = 1.0 / (1 + 0.05 * distance)
        else:
            # staying above target height is better
            reward = 1.0 / (1 + 0.01 * distance)
        # penalize crashes to keep the drone flying
        if done and self.sim.time < self.sim.runtime:
            reward -= 10
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        state_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(done)
            state = self.get_state()
            state_all.append(state)
        next_state = np.concatenate(state_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.get_state()] * self.action_repeat)
        return state

