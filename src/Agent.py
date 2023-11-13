import torch
import numpy as np
from src.DQN import DQN
from src.utils import ReplayMemory
from collections import namedtuple


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)

class Agent():
    def __init__(
        self, 
        n_observations : int, 
        n_actions : int,
        device : str,
        epsilon_decay : float = 1 - 1e-3,
        batch_size : int = 128
    ) -> None:
        
        self.policy_dqn = DQN(n_observations, n_actions).to(device)
        self.target_dqn = DQN(n_observations, n_actions).to(device)

        self.device = device

        self.epsilon = 0.9
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = 5e-2

        self.gamma = 1 - 1e-2

        self.n_actions = n_actions

        self.memory = ReplayMemory(1000)
        self.batch_size = batch_size

        self.optimizer = torch.optim.AdamW(
            self.policy_dqn.parameters(),
            1e-4   
        )

        self.t = 5e-3

    def act(self, state):

        state_tensor = torch.tensor(
                state,
                dtype= torch.float32,
                device=self.device,
        ).unsqueeze(0)



        if np.random.random() < self.epsilon:
            best_action = np.random.randint(0,self.n_actions)

        else:
            best_action = self.policy_dqn(state_tensor).argmax().item()

        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.epsilon_decay

        elif self.epsilon < self.final_epsilon:
            self.epsilon = self.final_epsilon

        return best_action
    
    def save_to_memory(
            self,
            state, 
            next_state, 
            action, 
            reward
        ):
            state_tensor = torch.tensor(
                state,
                dtype= torch.float32,
                device=self.device,
            ).unsqueeze(0)

            if next_state is not None:
                next_state_tensor = torch.tensor(
                    next_state,
                    dtype= torch.float32,
                    device=self.device,
                ).unsqueeze(0)

            else :
                next_state_tensor = None

            action_tensor = torch.tensor(
                [[action]],
                device = self.device,
                # dtype = torch.long
            )

            reward_tensor = torch.tensor(
                [reward],
                device = self.device
            )
            self.memory.push(
                state_tensor, 
                action_tensor, 
                next_state_tensor, 
                reward_tensor,
            )
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), 
                                            device=self.device, 
                                            dtype=torch.bool
                                    )
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_dqn(state_batch).gather(1, action_batch)

        
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_dqn(non_final_next_states).max(1)[0]

        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_dqn.parameters(), 100)
        self.optimizer.step()

        return loss
    
    def update_networks(self):

        target_net_state_dict = self.target_dqn\
            .state_dict()
        policy_net_state_dict = self.policy_dqn\
            .state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = \
                policy_net_state_dict[key]*self.t + target_net_state_dict[key]*(1-self.t)
            
        self.target_dqn.load_state_dict(target_net_state_dict)