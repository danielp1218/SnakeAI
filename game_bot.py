import torch
from torch import nn
import numpy as np
import random

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(11, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )


    def forward(self, x):
        return self.sequential(x)

class QTrainer:
    def __init__(self, model, lr, gamma, gamma_inflation=1.001, max_gamma=1):
        # Learning Rate for Optimizer
        self.lr = lr
        # Discount Rate
        self.gamma = gamma
        self.max_gamma = max_gamma
        self.g_inflation = gamma_inflation
        # Linear NN defined above.
        self.model = model
        # optimizer for weight and biases updation
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        # Mean Squared error loss function
        self.criterion = nn.MSELoss()

        if torch.cuda.is_available():
            torch.cuda.enable = True
            torch.device = "cuda"
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def train_step(self, statex, actionx, rewardx, next_statex, done):
        state = torch.tensor(statex).float()
        next_state = torch.tensor(next_statex).float()
        reward = torch.tensor(rewardx).float()
        action = torch.tensor(actionx).long()
        if isinstance(done, bool):
            state = [state]
            next_state = [next_state]
            reward = [reward]
            action = [action]
            done = [done]

        # 1. Predicted Q value with current state
        #print("action:", action)
        #print("reward:", reward)
        #print("next state:", next_state)
        for idx in range(len(done)):

            pred = self.model(state[idx])
            target = pred.clone().detach()

            if not done[idx]:
                target[action[idx]] = reward[idx]+ (self.gamma * self.model(next_state[idx]).argmax())
            else:
                target[action[idx]] = reward[idx]

            self.optimizer.zero_grad()
            loss = self.criterion(target, pred)
            loss.backward()  # backward propagation of loss
            #print("pred:",pred)
            #print("target:",target)
            self.optimizer.step()
            self.gamma = min(self.max_gamma, self.gamma*self.g_inflation)

        # 2. Q_new = reward + gamma * max(next_predicted Qvalue)
        # pred.clone()
        # preds[argmax(action)] = Q_new


class GameBot:
    def __init__(self, model_path=None, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, batch_size=32, lr=0.01, gamma=0.90, gamma_inflation=1.001, max_gamma=1):
        self.model = NeuralNetwork().float()
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = []
        self.batch_size = batch_size
        self.trainer = QTrainer(self.model, lr, gamma, gamma_inflation, max_gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if self.batch_size < len(self.memory):
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_move(self, state, smart_choose=False):
        if smart_choose and np.random.rand() <= self.epsilon:
            return self.model(torch.tensor(state).float()).topk(2).indices[1].item()
        if np.random.rand() <= self.epsilon:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            return np.random.randint(0, 3)
        return self.model(torch.tensor(state).float()).argmax().item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)
