import math

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
    


def policy(state):
    x, x_dot, theta, theta_dot = state
    z=params[0]*x + params[1]*x_dot + params[2]*theta + params[3]*theta_dot

    return 1 if z > 0 else 0      

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

def run_episode(env, params):

    state = env.reset()
    total_reward = 0

    while not env.is_done():

        x, x_dot, theta, theta_dot = state

        f = (
            params[0] * x
            + params[1] * x_dot
            + params[2] * theta
            + params[3] * theta_dot
        )



        state = env.step(f)
        draw_cartpole(state)

        total_reward += 1

    return total_reward
params = [50.0, 0.02, 0.5, 1.0]

current_reward = run_episode(env, params)

for iteration in range(100):

    candidate_params = params.copy()

    for i in range(4):
        candidate_params[i] += random.uniform(-0.1, 0.1)

    candidate_reward = run_episode(env, candidate_params)

    if candidate_reward > current_reward:

        params = candidate_params
        current_reward = candidate_reward

        print(
            "Accepted:",
            params,
            "reward:",
            current_reward
        )

plt.ioff()
plt.show()