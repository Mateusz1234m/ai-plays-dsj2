# TODO: comments

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
from PIL import Image
import sys

MAX_GAMES = 100
# MAX_MEMORY = MAX_GAMES * 10
MAX_MEMORY = 500
TRAINING_REPS = 3
MOVE = 10
BATCH_SIZE = 32
TAU = 1
LR = 3e-4
MOVE_PROBABILITIES = [100, 100, 100, 100]
N_ACTIONS = 4
TEST_JUMP_T = 5
UPDATE_FREQ = 2


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
        self.point_a = np.array([0, 0, 0.75])
        self.point_b = np.array([0, MAX_GAMES, 0])
        self.point_c = np.array([15, 0, 1])
        self.epsilon_a = 0
        self.epsilon_b = 0
        self.epsilon_c = 0
        self.lr = lr
        self.memory = deque(maxlen=MAX_MEMORY)
        self.memory_buffer = []

        self.online_model = neural_network_module.DQN(n_actions=1, device=self.device)
        self.target_model = neural_network_module.DQN(n_actions=1, device=self.device)

        self.online_model.load_state_dict(torch.load('models/resnet18_pretrained.pt'))
        self.target_model.load_state_dict(torch.load('models/resnet18_pretrained.pt'))

        self.online_model.resnet18.fc = nn.Linear(512, n_actions)
        self.target_model.resnet18.fc = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.target_model.eval()
        self.update_target()

    def calculate_probability_function(self):

        # calculate 2 vectors on a plane
        vec_1 = self.point_c - self.point_a
        vec_2 = self.point_b - self.point_a

        # calculate cross product of 2 vectors on a plane
        cross_product = np.cross(vec_1, vec_2)
        cross_product /= cross_product[2]

        temp = np.dot(cross_product, self.point_a)

        # print(f"{cross_product[0]}x + {cross_product[1]}y +{cross_product[2]}z + {-temp} = 0")
        # print(f"{cross_product[2]}z = {-cross_product[0]}x + {-cross_product[1]}y + {temp}")

        self.epsilon_a = -cross_product[0]
        self.epsilon_b = -cross_product[1]
        self.epsilon_c = temp

    def get_probability(self, iteration, epoch):
        return self.epsilon_a * iteration + self.epsilon_b * epoch + self.epsilon_c

    def update_target(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def save(self, filename):
        torch.save(self.online_model.state_dict(), filename)

    def load(self, filename):
        self.online_model.load_state_dict(torch.load(filename))
        # self.online_model.eval()

    def get_state(self, data_grabber):
        game_img = data_grabber.get_game_img()
        return np.asarray(game_img.resize((200, 116)), dtype=np.float32) / 255.0

    def remember(self, state, action, reward, next_state, game_over):
        self.memory_buffer.append([np.asarray(state), action, reward, np.asarray(next_state), game_over])

    def buffer_to_memory(self):
        for data in self.memory_buffer:
            self.memory.append(data)

        self.memory_buffer = []

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(np.array(state), dtype=torch.float32)
        # state = torchvision.transforms.ToTensor()(state)
        action = torch.tensor(action, dtype=torch.int64)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32)
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

        # print(reward)
        pred_qs = self.online_model(state)
        pred_qs = pred_qs.gather(1, action)

        target_qs = self.target_model(next_state)
        target_qs = torch.max(target_qs, dim=1).values
        target_qs = target_qs.reshape(-1, 1)
        target_qs[game_over] = 0.0

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

    def get_action(self, state, iteration, test_jump=False):

        """ Epsilon greedy policy """
        self.epsilon = MAX_GAMES - self.n_games
        #
        # self.epsilon = self.get_probability(iteration, epoch=self.n_games) * MAX_GAMES

        # self.epsilon = (2 * (((MAX_GAMES - self.n_games) / MAX_GAMES) - 0.5) + (iteration / 20)) * MAX_GAMES

        # if iteration <= 5:
        #     self.epsilon = MAX_GAMES / 4 - self.n_games
        # elif 5 < iteration <= 8:
        #     self.epsilon = MAX_GAMES / 2 - self.n_games
        # else:
        #     self.epsilon = MAX_GAMES - self.n_games

        # print(f"Random move probability: {self.epsilon/MAX_GAMES * 100 :.2f}")
        is_random = False
        if random.randint(0, MAX_GAMES) < self.epsilon and not test_jump:
            is_random = True
            move = random.randint(0, sum(MOVE_PROBABILITIES))
            if move >= 0 and move < MOVE_PROBABILITIES[0]:
                action = 0
            elif  move >= MOVE_PROBABILITIES[0] and move < MOVE_PROBABILITIES[0] + MOVE_PROBABILITIES[1]:
            # else:
                action = 1
            elif  move >= MOVE_PROBABILITIES[0] + MOVE_PROBABILITIES[1] and move < MOVE_PROBABILITIES[0] + MOVE_PROBABILITIES[1] + MOVE_PROBABILITIES[2]:
            # else:
                action = 2
            # elif  move >= MOVE_PROBABILITIES[0] + MOVE_PROBABILITIES[1] + MOVE_PROBABILITIES[2] and move < MOVE_PROBABILITIES[0] + MOVE_PROBABILITIES[1] + MOVE_PROBABILITIES[2] + MOVE_PROBABILITIES[3]:
            else:
                action = 3
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

        """ Boltzmann Q policy """
        # with torch.no_grad():
        #     state_0 = torch.unsqueeze(torch.tensor(np.asarray(state), dtype=torch.float32), 0)
        #     state_0 = torch.permute(state_0, (0, 3, 1, 2)).to(device=self.device)
        #     q_values = self.online_model(state_0)
        # # action = torch.argmax(prediction).item()
        # q_values = np.squeeze(np.asarray(q_values))
        # if not test_jump:
        #     tau = TAU
        #     probs = np.exp(q_values / tau) / np.sum(np.exp(q_values / tau))
        #     # print("###")
        #     # print(q_values)
        #     # print(probs)
        #     action = np.random.choice(a=N_ACTIONS, p=probs)
        #     is_random = True
        # else:
        #     # print("###")
        #     # print(q_values)
        #     action = np.argmax(q_values)
        #     is_random = False

        # print(action)

        return action, is_random


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
    agent.calculate_probability_function()
    # agent.load('best.pt')
    # agent.update_target()

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

    time.sleep(3)

    # uncomment the line below and comment line above to hardcode the jump moment,
    # remember to change if statement in line 298 to total_clicks == 1
    # and change comments in line 414 - 416

    # time.sleep(sleep_time)
    # game_controller.left_and_right_click()
    # time.sleep(0.5)

    # for i in range(4):
    #     game_controller.jumper_move(10)
    #     time.sleep(0.2)
    # time.sleep(5)
    global time_start
    global total_clicks
    global steps
    global iteration
    time_start = time.time()
    total_clicks = 0
    steps = 0
    iteration = 0
    while True:

        state_old = agent.get_state(data_grabber)

        if agent.n_games % TEST_JUMP_T == 0 and agent.n_games != 0:
            final_move, is_random = agent.get_action(state_old, iteration=iteration, test_jump=True)
            iteration += 1
        else:
            final_move, is_random = agent.get_action(state_old, iteration=iteration)
            iteration += 1

        game_over = data_grabber.get_game_over() or total_clicks == 2


        if not game_over:

            random_text = "RANDOM" if is_random else "TRAINED"
            if final_move == 0:
                game_controller.jumper_move(MOVE)
                print(f"{random_text}: DOWN")
            elif final_move == 1:
                print(f"{random_text}: pass")
                pass
            elif final_move == 2:
                game_controller.jumper_move(-MOVE)
                print(f"{random_text}: UP")
            elif final_move == 3:
                game_controller.left_and_right_click()
                total_clicks += 1
                print(f"{random_text}: BOTH CLICK")

            time.sleep(0.05)
            state_new = agent.get_state(data_grabber)

            # remember
            if not(agent.n_games % TEST_JUMP_T == 0 and agent.n_games != 0):
                agent.remember(state=state_old, action=final_move, reward=0, next_state=state_new, game_over=game_over)


        if game_over:

            points = None
            # print("Game Over")
            while points is None:
                # points_img = data_grabber.get_points_img()
                # points = points_classifier.get_points(points_img)

                distance_img = data_grabber.get_points_img()
                points = points_classifier.get_points(distance_img)

            # reset game
            points /= 10
            if points > record:
                record = points
                agent.save(filename='best.pt')

            # overwrite reward
            for data in agent.memory_buffer:
                data[2] = points

            if not (agent.n_games % TEST_JUMP_T == 0 and agent.n_games != 0):
                agent.buffer_to_memory()

            # print("Training long memory started")
            # for i in range(min(agent.n_games, 20)):
            if not (agent.n_games % TEST_JUMP_T == 0 and agent.n_games != 0):
                for i in range(TRAINING_REPS):
                    agent.train()
            # print("Training long memory finished")

            if not (agent.n_games % TEST_JUMP_T == 0 and agent.n_games != 0):

                if agent.n_games % agent.update_freq == 0:
                    agent.update_target()
                print(f"Iteration: {agent.n_games}, Score: {points * 10}, Best: {record * 10}")

            else:
                print(f"Test jump, Score: {points * 10}, Best: {record * 10}")
            agent.n_games += 1
            time.sleep(0.5)
            game_controller.left_click()
            time.sleep(0.5)
            game_controller.left_click()
            # game_controller.choose_hill('FINLAND')

            # print()
            time.sleep(3)
            # sleep_time += 0.01
            game_controller.left_click()

            # hardcoded or not
            time.sleep(3)
            # time.sleep(sleep_time)
            # game_controller.left_and_right_click()
            # time.sleep(0.5)

            # print(sleep_time)
            # for i in range(4):
            #     game_controller.jumper_move(10)
            #     time.sleep(0.2)
            # time.sleep(5)
            time_start = time.time()
            total_clicks = 0
            iteration = 0


if __name__ == '__main__':
    train()