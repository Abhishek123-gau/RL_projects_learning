import math
import random
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# ENVIRONMENT
# ============================================================

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


# ============================================================
# ACTOR
# ============================================================

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


# ============================================================
# CRITIC
# ============================================================

def calculate_value(state, params):

    x, x_dot, theta, theta_dot = state

    value = (
        params[0] * x
        + params[1] * x_dot
        + params[2] * theta
        + params[3] * theta_dot
    )

    return value


# ============================================================
# VISUALIZATION
# ============================================================

def draw_cartpole(state):

    x, x_dot, theta, theta_dot = state

    pole_length = 0.5

    pole_x = x + pole_length * math.sin(theta)
    pole_y = pole_length * math.cos(theta)

    plt.clf()

    plt.plot([-2.4, 2.4], [0, 0])

    plt.plot(
        [x - 0.2, x + 0.2],
        [0.1, 0.1],
        linewidth=5
    )

    plt.plot(
        [x, pole_x],
        [0.1, 0.1 + pole_y],
        linewidth=3
    )

    plt.xlim(-2.4, 2.4)
    plt.ylim(-0.1, 1.0)

    plt.pause(0.02)


# ============================================================
# A2C
# ============================================================

def run_episode(
    env,
    actor_params,
    critic_params,
    actor_lr=0.001,
    critic_lr=0.01,
    gamma=0.99
):

    state = env.reset()

    step = 0

    while not env.is_done():

        # ----------------------------------------------------
        # 1. COLLECT 4 STEPS
        # ----------------------------------------------------

        states = []
        actions = []
        rewards = []
        action_probabilities = []
        values = []

        for i in range(4):

            # Actor chooses action
            action, p_right = policy(
                state,
                actor_params
            )

            # Critic evaluates current state
            value = calculate_value(
                state,
                critic_params
            )

            # Convert discrete action to force
            if action == 0:
                force = -env.force_mag
            else:
                force = env.force_mag

            # Environment transition
            next_state = env.step(force)

            # Reward
            reward = (
                6 * math.pi / 180
                - abs(state[2])
            )

            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            action_probabilities.append(p_right)
            values.append(value)

            # Move forward
            state = next_state
            step += 1

            # Stop collecting if episode ended
            if env.is_done():
                break


        # ----------------------------------------------------
        # 2. BOOTSTRAP VALUE
        # ----------------------------------------------------

        if env.is_done():

            R = 0.0

        else:

            R = calculate_value(
                state,
                critic_params
            )


        # ----------------------------------------------------
        # 3. CALCULATE CUMULATIVE RETURNS
        # ----------------------------------------------------

        returns = [0.0] * len(rewards)

        for t in reversed(range(len(rewards))):

            R = rewards[t] + gamma * R

            returns[t] = R


        # ----------------------------------------------------
        # 4. CALCULATE ADVANTAGES
        # ----------------------------------------------------

        advantages = [0.0] * len(rewards)

        for t in range(len(rewards)):

            advantages[t] = (
                returns[t]
                - values[t]
            )


        # ----------------------------------------------------
        # 5. UPDATE ACTOR + CRITIC
        # ----------------------------------------------------

        for t in range(len(rewards)):

            state_array = np.array(
                states[t]
            )

            # ---------------- ACTOR ----------------

            actor_gradient = (
                actions[t]
                - action_probabilities[t]
            ) * state_array

            actor_params += (
                actor_lr
                * advantages[t]
                * actor_gradient
            )


            # ---------------- CRITIC ----------------

            critic_gradient = (
                advantages[t]
                * state_array
            )

            critic_params += (
                critic_lr
                * critic_gradient
            )


        draw_cartpole(state)


    return (
        step,
        actor_params,
        critic_params
    )


# ============================================================
# TRAINING
# ============================================================

env = CartPole()

actor_params = np.array(
    [0.0, 0.0, 0.0, 0.0],
    dtype=float
)

critic_params = np.array(
    [0.0, 0.0, 0.0, 0.0],
    dtype=float
)


for episode in range(500):

    step, actor_params, critic_params = run_episode(
        env,
        actor_params,
        critic_params
    )

    print(
        "Episode:",
        episode + 1,
        "Steps:",
        step,
        "Actor:",
        actor_params,
        "Critic:",
        critic_params
    )