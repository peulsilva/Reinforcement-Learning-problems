import torch
import numpy as np
from src.DQN import DQN
from src.utils import ReplayMemory
from collections import namedtuple


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)
import torch
import numpy as np
from src.DQN import DQN
from src.utils import ReplayMemory
from collections import namedtuple, defaultdict
from abc import abstractmethod


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)

class DQAgent():
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
        self.final_epsilon = 1e-3

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

class BaseAgent():
    def __init__(self,
                 n_observations : int,
                 n_actions : int,
                 alpha : float = 0.7,
                 epsilon_decay : float = 1-1e-4,
                 gamma : float = 0.95,
                 final_epsilon : float = 0.1,
                 
                 ) -> None:
        
        self.epsilon = final_epsilon  
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.alpha = alpha
        self.gamma = gamma
        self.q = defaultdict(lambda : np.zeros(n_actions))
        self.n_actions = n_actions

    def act(self, state):
        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.epsilon_decay

        elif self.epsilon < self.final_epsilon:
            self.epsilon = self.final_epsilon

        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        
        else:
            return np.argmax(self.q[state])
        
    @abstractmethod
    def learn(
        self, 
        state : tuple , 
        next_state : tuple, 
        action: int, 
        reward : int,
        next_action : int = None
    ):
        ...


class QAgent(BaseAgent):
    def __init__(self, 
                 n_observations : int, 
                 n_actions: int, 
                 alpha : float = 0.7,
                 epsilon_decay : float = 1-1e-4,
                 gamma : float = 0.95,
                 final_epsilon : float = 0.1
                ) -> None:
        
        super().__init__(n_observations, n_actions, alpha, epsilon_decay, gamma, final_epsilon)

    def learn(
        self, 
        state:tuple , 
        next_state: tuple, 
        action: int, 
        reward :int,
        next_action = None,
        terminated : bool = None,
    ):

        Q_old = self.q[state][action]
        max_Q_next = (not terminated) * np.max(self.q[next_state])
        
        Q_new = (1 - self.alpha) * Q_old + self.alpha * (reward +  self.gamma * max_Q_next)
        self.q[state][action] = Q_new

        delta =  reward + self.gamma * self.q[next_state][next_action] - self.q[state][action]
        return delta**2

class SarsaAgent(BaseAgent):
    def __init__(self, 
                 n_observations : int,
                 n_actions: int, 
                 alpha : float = 0.7,
                 epsilon_decay : float = 1-1e-4,
                 gamma : float = 0.95,
                 final_epsilon : float = 0.1
                ) -> None:
        
        super().__init__(n_observations, n_actions, alpha, epsilon_decay, gamma, final_epsilon)

    def learn(
        self, 
        state:tuple , 
        next_state: tuple, 
        action: int, 
        reward :int,
        next_action : int = None,
        terminated : bool = None
    ):
        self.q[state][action] += self.alpha * (reward + self.gamma * (not terminated )* self.q[next_state][next_action] - self.q[state][action])

        delta =  reward + self.gamma * self.q[next_state][next_action] - self.q[state][action]
        return delta**2
    
class MCAgent(BaseAgent):
    def __init__(self, 
                 n_observations : int,
                 n_actions: int, 
                 alpha : float = 0.7,
                 epsilon_decay : float = 1-1e-4,
                 gamma : float = 0.95,
                 final_epsilon : float = 0.1
                ) -> None:
        
        super().__init__(n_observations, n_actions, alpha, epsilon_decay, gamma, final_epsilon)
        self.state_count = defaultdict(float)

    def learn(self,
              state,
              action,
              reward,
              expected_reward,
              s_a_count):
        
        expected_reward = reward + self.gamma * expected_reward 
        
        self.q[state][action] = self.q[state][action] + (expected_reward - self.q[state][action]) / s_a_count

        return expected_reward
    
    def get_epsilon(self, state_count : int):
        return (100)/(100+state_count)
    
    def act(self, state):
        self.state_count[state]+=1 
        epsilon = self.get_epsilon(self.state_count[state])

        if np.random.rand() < epsilon:
            return np.random.randint(0, self.n_actions)
        
        else:
            return np.argmax(self.q[state])

    
        