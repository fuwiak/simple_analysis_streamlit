import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


class LogisticsEnv:
    def __init__(self, data, reward_column):
        self.data = data
        self.n_steps = data.shape[0]
        self.current_step = None
        self.done = None
        self.total_rewards = None
        self.reward_column = reward_column
        self.reset()

    def reset(self):
        self.current_step = 0
        self.done = False
        self.total_rewards = 0
        return self.current_step

    def step(self, action):
        reward = -self.data.iloc[self.current_step][self.reward_column]
        self.total_rewards += reward
        self.current_step += 1
        if self.current_step >= self.n_steps:
            self.done = True
            self.current_step = self.n_steps - 1
        return self.current_step, reward, self.done, {}


class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state_index):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state_index])

    def update_q_table(self, state_index, action, reward, next_state_index,
                       done):
        best_next_action = np.argmax(
            self.q_table[next_state_index]) if not done else 0
        td_target = reward + self.gamma * self.q_table[next_state_index][
            best_next_action]
        td_error = td_target - self.q_table[state_index][action]
        self.q_table[state_index][action] += self.lr * td_error


# Load default dataset
@st.cache(allow_output_mutation=True)
def load_default_data():
    path = 'data/merged_leads_land_not_null.csv'
    return pd.read_csv(path)


df = load_default_data()

# Streamlit UI setup
st.title("RL Model for Logistics Optimization")
reward_column = st.selectbox("Select Reward Column", df.columns,
                             index=df.columns.get_loc("price_usd"))
env = LogisticsEnv(df, reward_column=reward_column)
agent = QLearningAgent(n_states=env.n_steps, n_actions=3)

# Parameter configuration
n_episodes = st.sidebar.number_input("Number of Episodes", min_value=1,
                                     max_value=1000, value=100)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01,
                                  max_value=0.9, value=0.1)
discount_factor = st.sidebar.slider("Discount Factor", min_value=0.01,
                                    max_value=1.0, value=0.99)
epsilon = st.sidebar.slider("Exploration Rate (Epsilon)", min_value=0.01,
                            max_value=1.0, value=0.1)

# Update agent parameters
agent.lr = learning_rate
agent.gamma = discount_factor
agent.epsilon = epsilon

# Training button
if st.button('Train Agent'):
    total_rewards = []
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)

    rewards_df = pd.DataFrame(
        {"Episode": range(n_episodes), "Total Reward": total_rewards})
    fig = px.line(rewards_df, x="Episode", y="Total Reward",
                  title="Rewards Over Episodes")
    st.plotly_chart(fig)

# Reset environment and agent
if st.button('Reset Environment and Agent'):
    env = LogisticsEnv(df, reward_column=reward_column)
    agent = QLearningAgent(n_states=env.n_steps, n_actions=3, lr=learning_rate,
                           gamma=discount_factor, epsilon=epsilon)
    st.write("Environment and agent have been reset.")

# Display agent's learned Q-values
if st.button("Display Q-Values"):
    st.write(pd.DataFrame(agent.q_table,
                          columns=["Action 1", "Action 2", "Action 3"]))
