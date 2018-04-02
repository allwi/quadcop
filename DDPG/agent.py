import numpy as np
from DDPG.actorcritic import DDPGActor, DDPGCritic
from DDPG.tools import OUNoise, ReplayBuffer

class DDPGAgent():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        # Actor (Policy) Model
        self.actor = DDPGActor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = DDPGActor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic = DDPGCritic(self.state_size, self.action_size)
        self.critic_target = DDPGCritic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic.model.get_weights())
        self.actor_target.model.set_weights(self.actor.model.get_weights())

        # Noise process
        self.noise_mean = 0
        self.noise_decay = 0.1
        self.noise_variance = 5
        self.noise = OUNoise(self.action_size, self.noise_mean, self.noise_decay, self.noise_variance)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 16
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.1  # for soft update of target parameters

        self.best_score = -np.inf
        self.num_steps = 0

    def reset_episode(self):
        if self.get_score() > self.best_score:
            self.best_score = self.get_score()

        self.noise.reset()
        self.last_state = self.task.reset()

        self.total_reward = 0.0
        self.num_steps = 0

        return self.last_state

    def get_score(self):
        return -np.inf if self.num_steps == 0 else self.total_reward / self.num_steps

    def step(self, action, reward, next_state, done):
        self.total_reward += reward
        self.num_steps += 1
        
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(states, [-1, self.state_size])
        action = self.actor.model.predict(state)[0]
        action = list(action + self.noise.sample())  # add some noise for exploration
        return action

    def learn(self, experiences):
        # print('Learning phase')
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic.model, self.critic_target.model)
        self.soft_update(self.actor.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)