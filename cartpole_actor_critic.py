import math
import numpy as np
import matplotlib.pyplot as plt
params=[50.0, 0.02, 0.5, 1.0]
class CartPole:

    def __init__(self):

        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1

        self.total_mass = (
            self.mass_cart + self.mass_pole
        )

        self.length = 0.5
        self.force_mag = 50.0
        self.dt = 0.02

        self.reset()

    def reset(self):

        self.x = 0.0
        self.x_dot = 0.0

        self.theta = 0.05
        self.theta_dot = 0.0

        return self.get_state()

    def get_state(self):

        return [
            self.x,
            self.x_dot,
            self.theta,
            self.theta_dot
        ]
    def is_done(self):

        if abs(self.x) > 2.4:
            return True

        if abs(self.theta) > 12 * math.pi / 180:
            return True

        return False


    def step(self, force):

        force = max(
            -self.force_mag,
            min(self.force_mag, force)
        )
        sin_theta = math.sin(self.theta)
        cos_theta = math.cos(self.theta)

        temp = (
            force
            + self.mass_pole
            * self.length
            * self.theta_dot ** 2
            * sin_theta
        ) / self.total_mass

        theta_acc = (
            self.gravity * sin_theta
            - cos_theta * temp
        ) / (
            self.length *
            (
                4 / 3
                - self.mass_pole
                * cos_theta ** 2
                / self.total_mass
            )
        )

        x_acc = temp - (
            self.mass_pole
            * self.length
            * theta_acc
            * cos_theta
            / self.total_mass
        )

        self.x_dot += self.dt * x_acc
        self.theta_dot += self.dt * theta_acc

        self.x += self.dt * self.x_dot
        self.theta += self.dt * self.theta_dot

        return self.get_state()
    


def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def actor_policy(state, params):

    x, x_dot, theta, theta_dot = state

    z = (
        params[0] * x
        + params[1] * x_dot
        + params[2] * theta
        + params[3] * theta_dot
    )

    p_right = sigmoid(z)

    if np.random.rand() < p_right:
        action = 1       # RIGHT
    else:
        action = 0       # LEFT

    return action, p_right
def value_function(state, params):

    x, x_dot, theta, theta_dot = state

    z = (
        params[0] * x
        + params[1] * x_dot
        + params[2] * theta
        + params[3] * theta_dot
    )

    value = z

    return value

def criti_policy(state, params):

    x, x_dot, theta, theta_dot = state

    z = (
        params[0] * x
        + params[1] * x_dot
        + params[2] * theta
        + params[3] * theta_dot
    )

    p_right = sigmoid(z)

    if np.random.rand() < p_right:
        action = 1       # RIGHT
    else:
        action = 0       # LEFT

    return action, p_right

def policy(state, params):

    x, x_dot, theta, theta_dot = state

    z = (
        params[0] * x
        + params[1] * x_dot
        + params[2] * theta
        + params[3] * theta_dot
    )

    p_right = sigmoid(z)

    if random.random() < p_right:
        action = 1       # RIGHT
    else:
        action = 0       # LEFT

    return action, p_right

# import math

def draw_cartpole(state):

    x, x_dot, theta, theta_dot = state

    # Pole length
    pole_length = 0.5

    # Pole endpoint
    pole_x = x + pole_length * math.sin(theta)
    pole_y = pole_length * math.cos(theta)

    plt.clf()

    # Ground
    plt.plot([-2.4, 2.4], [0, 0])

    # Cart
    plt.plot(
        [x - 0.2, x + 0.2],
        [0.1, 0.1],
        linewidth=5
    )

    # Pole
    plt.plot(
        [x, pole_x],
        [0.1, 0.1 + pole_y],
        linewidth=3
    )

    plt.xlim(-2.4, 2.4)
    plt.ylim(-0.1, 1.0)

    plt.pause(0.02)
env = CartPole()

state = env.reset()

# plt.ion()
# old_reward = 0.0
# for _ in range(100):
#     reward = 0.0
#     state = env.reset()
#     while not env.is_done():
        

#         action = policy(state)

#         state = env.step(action)
#         reward += 1.0
#         draw_cartpole(state)
#     print("Reward:", reward)
#     if(reward>old_reward):
#         old_reward=reward
#         params[0]-=0.1
#         params[1]-=0.1
#         params[2]-=0.1
#         params[3]-=0.1



import random
def sigmoid(z):
    return 1 / (1 + math.exp(-z))
def policy(state, params):

    x, x_dot, theta, theta_dot = state

    z = (
        params[0] * x
        + params[1] * x_dot
        + params[2] * theta
        + params[3] * theta_dot
    )

    p_right = sigmoid(z)

    if random.random() < p_right:
        action = 1       # RIGHT
    else:
        action = 0       # LEFT

    return action, p_right
def calculate_value(state, params):

    x, x_dot, theta, theta_dot = state

    z = (
        params[0] * x
        + params[1] * x_dot
        + params[2] * theta
        + params[3] * theta_dot
    )

    value = z

    return value
def run_episode(env, actor_params, critic_params, actor_lr=0.001, critic_lr=0.01, gama=0.99):

    state = env.reset()
    step=0

    while not env.is_done():
        action, p_right = policy(state, actor_params)
        value = calculate_value(state, critic_params)
        next_state = env.step(action)
        after_value = calculate_value(next_state, critic_params)
        reward = (6 * math.pi / 180)-state[2]
        y=reward+gama*after_value
        td_error=y-value
        # print("td_error:", td_error)
        actor_gradient=(action - p_right) * np.array(state)
        actor_params+= actor_lr * td_error * actor_gradient
        critic_gradient=np.array(state)
        critic_params+= critic_lr * td_error * critic_gradient
        draw_cartpole(next_state)
        state=next_state
        step+=1
        # rewards.append(reward)
        # print("Reward:", reward)
    
        

    return step, actor_params, critic_params
actor_params = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
critic_params = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
for episode in range(500):

    step, actor_params, critic_params = run_episode(env, actor_params, critic_params)
    print("Episode:", episode + 1, "Steps:", step, "Actor Params:", actor_params, "Critic Params:", critic_params)

