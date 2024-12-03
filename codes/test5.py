import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
import random
from datetime import datetime, timedelta

# Step 1: Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data['system_id'] = data['system_id'].astype(int)
    data['sentiment'] = data['comment'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return data

# Step 2: Environment class
class FailurePredictionEnv:
    def __init__(self, data):
        self.data = data
        self.fail_types = data['fail_type'].unique()
        self.label_encoder = LabelEncoder()
        self.data['fail_type_encoded'] = self.label_encoder.fit_transform(data['fail_type'])
        self.num_actions = len(self.fail_types)
        self.state_space = ['system_id', 'sentiment']
        self.action_space = range(self.num_actions)

    def get_recent_state(self, system_id, input_date):
        # Filter comments for the given system and recent period (e.g., last 7 days)
        recent_data = self.data[
            (self.data['system_id'] == system_id) &
            (self.data['date'] >= input_date - timedelta(days=7))
        ]
        if recent_data.empty:
            return None  # No relevant state available

        # Aggregate sentiment and return the state
        mean_sentiment = recent_data['sentiment'].mean()
        return np.array([system_id, mean_sentiment])

    def get_action_name(self, action):
        return self.label_encoder.inverse_transform([action])[0]

# Step 3: Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_size, num_actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.state_size = state_size
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}

    def _state_to_key(self, state):
        return tuple(state)

    def act(self, state):
        state_key = self._state_to_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)

        if np.random.rand() < self.epsilon:
            return random.choice(range(self.num_actions))  # Explore
        else:
            return np.argmax(self.q_table[state_key])  # Exploit

    def update(self, state, action, reward, next_state):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.num_actions)

        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.gamma * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.alpha * td_error

# Step 4: Train the Q-Learning Agent
def train_agent(env, agent, episodes=500):
    for episode in range(episodes):
        state = env.get_recent_state(random.choice(env.data['system_id']), datetime.now())
        if state is None:
            continue
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state = state  # Static environment; update for dynamic cases
            reward = random.choice([-1, 1])  # Placeholder reward; replace with logic
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            done = random.choice([True, False])  # Placeholder; replace with termination logic

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Step 5: Predict Fail Type or Success
def predict_failure(env, agent, system_id, input_date):
    input_date = pd.to_datetime(input_date)
    state = env.get_recent_state(system_id, input_date)

    if state is None:
        print(f"No recent data found for System {system_id}. Unable to predict.")
        return

    action = agent.act(state)
    prediction = env.get_action_name(action)
    print(f"Predicted Failure Type for System {system_id} on {input_date.date()}: {prediction}")

# Main Execution
if __name__ == "__main__":
    # Load the dataset
    file_path = "comments.csv"  # Replace with your file path
    data = load_data(file_path)

    # Initialize the environment
    env = FailurePredictionEnv(data)

    # Initialize the agent
    agent = QLearningAgent(state_size=len(env.state_space), num_actions=env.num_actions)

    # Train the agent
    print("Training the agent...")
    train_agent(env, agent, episodes=500)

    # Predict failure type for a specific system ID and date
    system_id = 101  # Example system ID
    input_date = "2024-11-06"  # Example date
    print("\nPredicting failure type...")
    predict_failure(env, agent, system_id, input_date)