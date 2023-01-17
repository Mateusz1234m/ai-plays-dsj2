import random
import torch
import acquisition_module
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import game_control_module
import neural_network_module
import time
import sys

MAX_MEMORY = 500
MAX_GAMES = 200
BATCH_SIZE = 32
LR = 0.0001
# MOVE_PROBABILITIES = [10, 10, 100, 100, 100]
# MOVE_PROBABILITIES = [10, 100, 100, 100]
MOVE_PROBABILITIES = [100, 100, 100]
N_ACTIONS = 3
UPDATE_FREQ = 4


class Agent:
    """
    Agent class.
    """
    def __init__(self, n_actions=N_ACTIONS, update_freq=UPDATE_FREQ, lr=LR):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_games = 0
        self.n_actions = n_actions
        self.n_random_moves = 0
        self.n_trained_moves = 0
        self.update_freq = update_freq
        self.epsilon = 0
        self.gamma = 0.9
        self.lr = lr
        self.memory = deque(maxlen=MAX_MEMORY)
        self.memory_buffer = []
        self.online_model = neural_network_module.DQN(n_actions=self.n_actions, device=self.device)
        self.target_model = neural_network_module.DQN(n_actions=self.n_actions, device=self.device)
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.target_model.eval()
        self.update_target()
        # self.trainer = neural_network_module.DQNTrainer(model=self.online_model, lr=LR, gamma=self.gamma, device=self.device)

    def update_target(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def save(self, filename):
        torch.save(self.online_model.state_dict(), filename)

    def load(self, filename):
        self.online_model.load_state_dict(torch.load(filename))
        self.online_model.eval()

    def get_state(self, data_grabber):
        game_img = data_grabber.get_game_img()
        return game_img

    def remember(self, state, action, reward, next_state, game_over):
        self.memory_buffer.append([np.asarray(state), action, reward, np.asarray(next_state), game_over])

    def buffer_to_memory(self):
        for data in self.memory_buffer:
            self.memory.append(data)

        self.memory_buffer = []

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(np.asarray(state)/255.0, dtype=torch.float32)
        # state = torchvision.transforms.ToTensor()(state)
        action = torch.tensor(action, dtype=torch.int64)
        next_state = torch.tensor(np.asarray(next_state)/255.0, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        game_over = torch.tensor(game_over, dtype=torch.int64)

        if len(state.shape) == 3:
            state = torch.unsqueeze(state, 0).to(device=self.device)
            action = torch.unsqueeze(action, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = torch.unsqueeze(game_over, 0)

        state = torch.permute(state, (0, 3, 1, 2))
        next_state = torch.permute(next_state, (0, 3, 1, 2))

        action = action.reshape(-1, 1)
        reward = reward.reshape(-1, 1)
        game_over = game_over.reshape(-1, 1)

        pred_qs = self.online_model(state)
        pred_qs = pred_qs.gather(1, action)

        target_qs = self.target_model(next_state)
        target_qs = torch.max(target_qs, dim=1).values
        target_qs = target_qs.reshape(-1, 1)
        target_qs[game_over]=0.0

        y_js = reward + (self.gamma * target_qs)
        self.optimizer.zero_grad()
        loss = self.criterion(pred_qs, y_js)
        print(f"Loss: {loss.item()}")
        loss.backward()
        self.optimizer.step()

    def train(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, game_overs)

    def get_action(self, state):
        self.epsilon = MAX_GAMES - self.n_games
        final_move = [0]*self.n_actions
        if random.randint(0, MAX_GAMES) < self.epsilon:
            move = random.randint(0, sum(MOVE_PROBABILITIES))
            if move >= 0 and move < MOVE_PROBABILITIES[0]:
                action = 0
            elif  move >= MOVE_PROBABILITIES[0] and move < MOVE_PROBABILITIES[0] + MOVE_PROBABILITIES[1]:
            # else:
                action = 1
            # elif  move >= MOVE_PROBABILITIES[0] + MOVE_PROBABILITIES[1] and move < MOVE_PROBABILITIES[0] + MOVE_PROBABILITIES[1] + MOVE_PROBABILITIES[2]:
            else:
                action = 2
            # elif  move >= MOVE_PROBABILITIES[0] + MOVE_PROBABILITIES[1] + MOVE_PROBABILITIES[2] and move < MOVE_PROBABILITIES[0] + MOVE_PROBABILITIES[1] + MOVE_PROBABILITIES[2] + MOVE_PROBABILITIES[3]:
            #     action = 3
            # elif  move >= MOVE_PROBABILITIES[0] + MOVE_PROBABILITIES[1] + MOVE_PROBABILITIES[2] + MOVE_PROBABILITIES[3] and move <= MOVE_PROBABILITIES[0] + MOVE_PROBABILITIES[1] + MOVE_PROBABILITIES[2] + MOVE_PROBABILITIES[3] + MOVE_PROBABILITIES[4]:
            #     action = 4

            self.n_random_moves += 1
            # print(f'Random move {self.n_random_moves}, epsilon = {self.epsilon}')

        else:
            with torch.no_grad():
                state_0 = torch.unsqueeze(torch.tensor(np.asarray(state), dtype=torch.float32), 0)
                state_0 = torch.permute(state_0, (0, 3, 1, 2)).to(device=self.device)
                prediction = self.online_model(state_0)
            action = torch.argmax(prediction).item()

            # final_move[move] = 1
            self.n_trained_moves += 1
            # print(f'Trained move {self.n_trained_moves}, epsilon = {self.epsilon}')

        return action


def train():
    """
    Train function.
    :return:
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = -999
    agent = Agent()
    game_controller = game_control_module.GameController()
    data_grabber = acquisition_module.DataGrabber()
    points_classifier = neural_network_module.PointsClassifier()

    game_controller.focus_game()
    game_controller.choose_training()
    game_controller.choose_hill('FINLAND')

    time.sleep(3)
    game_controller.left_click()
    global sleep_time
    sleep_time = 3.74
    time.sleep(sleep_time)
    game_controller.left_and_right_click()
    time.sleep(0.5)
    # for i in range(4):
    #     game_controller.jumper_move(10)
    #     time.sleep(0.2)
    # time.sleep(5)
    global time_start
    global total_clicks
    global steps
    time_start = time.time()
    total_clicks = 0
    steps = 0
    while True:
        # time.sleep(3)
        # game_controller.left_click()
        state_old = agent.get_state(data_grabber)

        final_move = agent.get_action(state_old)

        game_over = data_grabber.get_game_over() or total_clicks == 1
        # perform move
        # print(final_move)

        if not game_over:
            # if final_move[0] == 1:
            #     pass
            #     # game_controller.left_click()
            #     # print('MOVE: Left click')
            #     # total_clicks += 1
            # elif final_move[1] == 1:
            #     game_controller.left_and_right_click()
            #     # print('MOVE: Both click')
            #     total_clicks += 1
            # elif final_move[2] == 1:
            #     game_controller.jumper_move(10)
            #     # print('MOVE: Jumper up')
            # elif final_move[3] == 1:
            #     game_controller.jumper_move(-10)
            #     # print('MOVE: Jumper down')
            # elif final_move[4] == 1:
            #     pass
            #     # print('MOVE: pass')

            # print(f"Action: {final_move}")
            if final_move == 0:
                game_controller.jumper_move(10)
                # print('MOVE: Jumper up')
                # game_controller.left_and_right_click()
                # print('MOVE: Both click')
                # total_clicks += 1
            if final_move == 1:
                game_controller.jumper_move(-10)
                # print('MOVE: Jumper down')
            # elif final_move == 1:
            #     game_controller.jumper_move(10)
            #     # print('MOVE: Jumper up')
            # elif final_move == 2:
            #     game_controller.jumper_move(-10)
            #     # print('MOVE: Jumper down')
            # elif final_move == 3:
            #     pass
            #     # print('MOVE: pass')

            elif final_move == 2:
                pass
                # print('MOVE: pass')

            state_new = agent.get_state(data_grabber)

            # remember
            agent.remember(state=state_old, action=final_move, reward=0, next_state=state_new, game_over=game_over)


        if game_over:

            points = None
            # print("Game Over")
            while points is None:
                points_img = data_grabber.get_points_img()
                points = points_classifier.get_points(points_img)

            # reset game
            points /= 100
            if points > record:
                record = points
                agent.save(filename='best.pt')

            # overwrite reward
            for data in agent.memory_buffer:
                data[2] = points

            agent.buffer_to_memory()

            # print("Training long memory started")
            agent.train()
            # print("Training long memory finished")

            agent.n_games += 1
            if agent.n_games % agent.update_freq == 0:
                agent.update_target()
            print(f"Iteration: {agent.n_games}, Score: {points * 100}, Best: {record * 100}")

            game_controller.left_click()
            game_controller.choose_hill('FINLAND')

            # print()
            time.sleep(3)
            # sleep_time += 0.01
            game_controller.left_click()
            time.sleep(sleep_time)
            game_controller.left_and_right_click()
            time.sleep(0.5)
            # print(sleep_time)
            # for i in range(4):
            #     game_controller.jumper_move(10)
            #     time.sleep(0.2)
            # time.sleep(5)
            time_start = time.time()
            total_clicks = 0


if __name__ == '__main__':
    train()