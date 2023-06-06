import pygame
import random
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import layers
from collections import deque

# Definição da interface gráfica

WIDTH = 800
HEIGHT = 400
SCORE_SIZE = 36

#Velocidades da bola
initial_ball_speed = 3
max_ball_speed = 5.0
acceleration_interval = 1.0
ball_acceleration = 0.01
iteration_count = 0
class Ball(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([10, 10])
        self.image.fill((255, 255, 255))
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.velocity = [random.choice([-1, 1]), random.choice([-1, 1])]


class Paddle(pygame.sprite.Sprite):
    def __init__(self, x):
        super().__init__()
        self.image = pygame.Surface([10, 60])
        self.image.fill((255, 255, 255))
        self.rect = self.image.get_rect()
        self.rect.center = (x, HEIGHT // 2)
        self.velocity = 0

    def update(self):
        self.rect.y += self.velocity
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT

def draw_elements(score1, score2):
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (255, 255, 255), [WIDTH // 2 - 1, 0, 2, HEIGHT])
    all_sprites.draw(screen)

    score_text = font.render(str(score1), True, (255, 255, 255))
    screen.blit(score_text, (WIDTH // 2 - 50, 10))

    score_text = font.render(str(score2), True, (255, 255, 255))
    screen.blit(score_text, (WIDTH // 2 + 30, 10))

# Definição dos agentes inteligentes


class DQNAgent:
    def __init__(self):
        self.action_space = [0, 1, 2]  # 0: não fazer nada, 1: mover raquete para cima, 2: mover raquete para baixo
        self.state_space = 4  # dimensões da observação do jogo

        self.epsilon = 0.90  # taxa de exploração inicial
        self.epsilon_decay = 0.995  # fator de decaimento da taxa de exploração
        self.epsilon_min = 0.01  # taxa de exploração mínima

        self.learning_rate = 0.01  # taxa de aprendizado
        self.discount_factor = 0.76  # fator de desconto

        self.memory = deque(maxlen=100000)  # memória de replay com capacidade máxima de 10.000 experiências


        self.model = self.build_model()  # construção do modelo da rede neural

    def build_model(self):
        model = keras.Sequential()
        model.add(layers.Dense(64, input_shape=(self.state_space,), activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(len(self.action_space), activation="linear"))
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space)
        q_values = self.model.predict(np.array([state]))
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.discount_factor * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
class RandomAgent:
    def __init__(self):
        self.action_space = [0, 1, 2]  # 0: não fazer nada, 1: mover raquete para cima, 2: mover raquete para baixo

    def act(self, state):
        return random.choice(self.action_space)

class RuleBasedAgent:
    def __init__(self):
        self.action_space = [0, 1, 2]  # 0: não fazer nada, 1: mover raquete para cima, 2: mover raquete para baixo

    def act(self, state):
        ball_y = state[1]
        paddle2_y = state[5]

        if paddle2_y < ball_y:
            return 2  # mover raquete para baixo
        elif paddle2_y > ball_y:
            return 1  # mover raquete para cima
        else:
            return 0  # não fazer nada
class HeuristicAgent:
    def __init__(self):
        self.action_space = [0, 1, 2]  # 0: não fazer nada, 1: mover raquete para cima, 2: mover raquete para baixo

    def act(self, state):
        ball_x, ball_y, ball_vx, ball_vy, paddle1_y, paddle2_y = state

        if ball_vx > 0:  # Somente faz a previsão se a bola estiver se movendo em direção ao paddle2
            # Calcula o tempo necessário para a bola chegar à altura do paddle2
            time_to_reach_paddle = (WIDTH - ball_x) / ball_vx

            # Calcula a posição vertical prevista da bola no momento em que ela atingir a altura do paddle2
            predicted_ball_y = ball_y + ball_vy * time_to_reach_paddle

            if predicted_ball_y < paddle2_y:
                return 1  # mover raquete para cima
            elif predicted_ball_y > paddle2_y:
                return 2  # mover raquete para baixo

        return 0  # não fazer nada

#codigo principal
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")


clock = pygame.time.Clock()
font = pygame.font.Font(None, SCORE_SIZE)

all_sprites = pygame.sprite.Group()
paddles = pygame.sprite.Group()
ball = Ball()
paddle1 = Paddle(10)
paddle2 = Paddle(WIDTH - 10)

all_sprites.add(ball, paddle1, paddle2)
paddles.add(paddle1, paddle2)

score1 = 0
score2 = 0

agent_dqn = DQNAgent()
agent_random = RandomAgent()
agent_rulebased = RuleBasedAgent()
agent_heuristic = HeuristicAgent()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = np.array([ball.rect.centerx, ball.rect.centery, ball.velocity[0], ball.velocity[1], paddle1.rect.centery, paddle2.rect.centery])
    if ball.rect.centery < paddle1.rect.centery:
        paddle1.velocity = -5 
    elif ball.rect.centery > paddle1.rect.centery:
        paddle1.velocity = 5  
    else:
        paddle1.velocity = 0  

    paddle1.update()

    action_dqn = agent_dqn.act(state)
    action_random = agent_random.act(state)
    action_rulebased = agent_rulebased.act(state)
    action_heuristic = agent_heuristic.act(state)

    paddle1.velocity = 0
    if action_dqn == 1:
        paddle1.velocity = -5
    elif action_dqn == 2:
        paddle1.velocity = 5

    paddle1.update()

    paddle2.velocity = 0
    if action_random == 1:
        paddle2.velocity = -5
    elif action_random == 2:
        paddle2.velocity = 5
    if action_heuristic == 1:
        paddle2.velocity = -5
    elif action_heuristic == 2:
        paddle2.velocity = 5
    paddle2.update()

    ball.rect.x += ball.velocity[0]
    ball.rect.y += ball.velocity[1]

    if ball.rect.colliderect(paddle1.rect):
        ball.velocity[0] = abs(ball.velocity[0])  # Inverte a direção horizontal
        reward = 10  # Define a recompensa positiva
        # Verificar colisão com o paddle 2
    elif ball.rect.colliderect(paddle2.rect):
        ball.velocity[0] = -abs(ball.velocity[0])  # Inverte a direção horizontal
        reward = 10  # Define a recompensa positiva

    if iteration_count % acceleration_interval == 0:
        if abs(ball.velocity[0]) < max_ball_speed:
            ball.velocity[0] += np.sign(ball.velocity[0]) * ball_acceleration
        if abs(ball.velocity[1]) < max_ball_speed:
            ball.velocity[1] += np.sign(ball.velocity[1]) * ball_acceleration

    if abs(ball.velocity[0]) > max_ball_speed:
        ball.velocity[0] = max_ball_speed * np.sign(ball.velocity[0])
    if abs(ball.velocity[1]) > max_ball_speed:
        ball.velocity[1] = max_ball_speed * np.sign(ball.velocity[1])

    if ball.rect.y >= HEIGHT - 10 or ball.rect.y <= 0:
        ball.velocity[1] = -ball.velocity[1]

    if ball.rect.x >= WIDTH - 10:
        score1 += 1
        reward = -20
        done = True
        ball.rect.center = (WIDTH // 2, HEIGHT // 2)
        ball.velocity = [3, 3]
    elif ball.rect.x <= 0:
        score2 += 1
        reward = 20
        done = True
        ball.rect.center = (WIDTH // 2, HEIGHT // 2)
        ball.velocity = [-3, -3]
    else:
        reward = 0
        done = False
    iteration_count += 1

    next_state = np.array([ball.rect.centerx, ball.rect.centery, ball.velocity[0], ball.velocity[1], paddle1.rect.centery, paddle2.rect.centery])
    agent_dqn.remember(state, action_dqn, reward, next_state, done)
    state = next_state


    ball.velocity[0] *= 1.01
    ball.velocity[1] *= 1.01

    draw_elements(score1, score2)
    pygame.display.flip()
    clock.tick(60)

    episode_count = 10000  # número de episódios de treinamento
    batch_size = 32  # tamanho do lote para o replay

for episode in range(episode_count):
    state = np.array([ball.rect.centerx, ball.rect.centery, ball.velocity[0], ball.velocity[1], paddle1.rect.centery, paddle2.rect.centery])
    done = False
    score1 = 0
    score2 = 0

    while not done:
        action_dqn = agent_dqn.act(state)
        action_random = agent_random.act(state)
        action_rulebased = agent_rulebased.act(state)

        paddle1.velocity = 0
        if action_dqn == 1:
            paddle1.velocity = -5
        elif action_dqn == 2:
            paddle1.velocity = 5
            # Ajustando a velocidade do Paddle1
         

            paddle1.update()

            paddle2.velocity = 0
            if action_rulebased == 1:
                paddle2.velocity = -5
            elif action_rulebased == 2:
                paddle2.velocity = 5
            paddle2.update()

        ball.rect.x += ball.velocity[0]
        ball.rect.y += ball.velocity[1]

        if ball.rect.y >= HEIGHT - 10 or ball.rect.y <= 0:
            ball.velocity[1] = -ball.velocity[1]
        if pygame.sprite.collide_rect(ball, paddle1) or pygame.sprite.collide_rect(ball, paddle2):
            ball.velocity[0] = -ball.velocity[0]

        if ball.rect.x >= WIDTH - 10:
            score1 += 1
            reward = -20
            done = True
            ball.rect.center = (WIDTH // 2, HEIGHT // 2)
        elif ball.rect.x <= 0:
            score2 += 1
            reward = 20
            done = True
            ball.rect.center = (WIDTH // 2, HEIGHT // 2)
        else:
            reward = 0
            done = False

        next_state = np.array([ball.rect.centerx, ball.rect.centery, ball.velocity[0], ball.velocity[1], paddle1.rect.centery, paddle2.rect.centery])
        agent_dqn.remember(state, action_dqn, reward, next_state, done)
        state = next_state

        draw_elements(score1, score2)
        pygame.display.flip()
        clock.tick(60)

    if len(agent_dqn.memory) > batch_size:
        agent_dqn.replay(batch_size)

pygame.quit()
