# 3. Implement Deep Q-Network (DQN) Agent for Firewall Policy Learning
import random
from collections import deque
import numpy as np
import json

class DQNFirewallAgent:
    """Deep Q-Network Agent for Dynamic Firewall Rule Optimization"""
    
    def __init__(self, state_size=10, action_size=4, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size  # ALLOW, BLOCK, RATE_LIMIT, LOG
        self.learning_rate = learning_rate
        
        # Hyperparameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Network parameters (simplified)
        self.q_table = {}  # Simplified Q-table instead of neural network
        
        # Action mapping
        self.actions = {
            0: 'ALLOW',
            1: 'BLOCK', 
            2: 'RATE_LIMIT',
            3: 'LOG_ONLY'
        }
        
        # Performance tracking
        self.training_rewards = []
        self.episode_count = 0
        
    def get_state_key(self, state):
        """Convert state array to string key for Q-table"""
        return str(tuple(np.round(state, 2)))
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def calculate_reward(self, action, traffic_type, detection_correct):
        """Calculate reward based on action effectiveness"""
        base_reward = 0
        
        # Reward structure based on action appropriateness
        if traffic_type == 'normal':
            if action == 0:  # ALLOW normal traffic
                base_reward = 10 if detection_correct else -5
            elif action == 1:  # BLOCK normal traffic (false positive)
                base_reward = -20
            elif action == 2:  # RATE_LIMIT normal traffic
                base_reward = -5
            else:  # LOG_ONLY normal traffic
                base_reward = 0
        else:  # Attack traffic
            if action == 0:  # ALLOW attack traffic (false negative)
                base_reward = -30
            elif action == 1:  # BLOCK attack traffic
                base_reward = 20 if detection_correct else -10
            elif action == 2:  # RATE_LIMIT attack traffic
                base_reward = 10
            else:  # LOG_ONLY attack traffic
                base_reward = -10
        
        return base_reward
    
    def train(self, experiences):
        """Train the DQN agent using experience replay"""
        if len(experiences) < self.batch_size:
            return
        
        batch = random.sample(experiences, self.batch_size)
        total_loss = 0
        
        for state, action, reward, next_state, done in batch:
            state_key = self.get_state_key(state)
            next_state_key = self.get_state_key(next_state)
            
            # Initialize Q-values if not exist
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)
            
            # Q-learning update
            target = reward
            if not done:
                target += self.gamma * np.amax(self.q_table[next_state_key])
            
            current_q = self.q_table[state_key][action]
            self.q_table[state_key][action] = current_q + self.learning_rate * (target - current_q)
            
            total_loss += abs(target - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss / len(batch)

class FirewallEnvironment:
    """Simulated Firewall Environment for DQN Training"""
    
    def __init__(self, anomaly_detector):
        self.detector = anomaly_detector
        self.current_traffic = None
        self.performance_metrics = {
            'threats_blocked': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_packets': 0
        }
    
    def reset(self):
        """Reset environment to initial state"""
        # Generate random network state
        state = np.random.rand(10)  # 10-dimensional state space
        return state
    
    def step(self, action, traffic_sample, true_label):
        """Execute action and return next state, reward, done"""
        # Get prediction from anomaly detector
        if len(traffic_sample.shape) == 1:
            traffic_sample = traffic_sample.reshape(1, -1)
        
        # Create sequence for detector (simplified)
        sequence = np.tile(traffic_sample, (5, 1)).reshape(1, 5, -1)
        prediction, confidence = self.detector.predict(sequence)
        
        # Calculate reward
        detection_correct = (prediction[0] == true_label)
        traffic_type = 'attack' if true_label == 1 else 'normal'
        
        reward = self.calculate_environment_reward(action, traffic_type, detection_correct, confidence[0])
        
        # Update metrics
        self.update_metrics(action, true_label, prediction[0])
        
        # Generate next state
        next_state = np.random.rand(10)
        done = False  # Continuous environment
        
        return next_state, reward, done
    
    def calculate_environment_reward(self, action, traffic_type, detection_correct, confidence):
        """Calculate comprehensive reward considering multiple factors"""
        base_reward = 0
        
        # Action effectiveness
        if traffic_type == 'normal':
            if action == 0:  # ALLOW
                base_reward = 10
            elif action == 1:  # BLOCK (false positive penalty)
                base_reward = -15
            elif action == 2:  # RATE_LIMIT (minor penalty)
                base_reward = -3
            else:  # LOG_ONLY
                base_reward = 1
        else:  # attack
            if action == 0:  # ALLOW (false negative penalty)
                base_reward = -25
            elif action == 1:  # BLOCK (correct action)
                base_reward = 15
            elif action == 2:  # RATE_LIMIT (partial protection)
                base_reward = 8
            else:  # LOG_ONLY (insufficient action)
                base_reward = -5
        
        # Confidence bonus
        confidence_bonus = confidence * 2 if detection_correct else -confidence * 2
        
        return base_reward + confidence_bonus
    
    def update_metrics(self, action, true_label, predicted_label):
        """Update performance metrics"""
        self.performance_metrics['total_packets'] += 1
        
        if true_label == 1:  # Attack
            if action == 1:  # Blocked
                self.performance_metrics['threats_blocked'] += 1
            elif action == 0:  # Allowed (false negative)
                self.performance_metrics['false_negatives'] += 1
        else:  # Normal
            if action == 1:  # Blocked (false positive)
                self.performance_metrics['false_positives'] += 1

# Initialize DQN Agent and Environment
print("\n=== Deep Q-Network Firewall Agent ===")
agent = DQNFirewallAgent(state_size=10, action_size=4)
# Initialize a placeholder or mock anomaly detector
class MockAnomalyDetector:
    def predict(self, sequence):
        # Mock prediction: always predicts 'normal' (0) with high confidence
        return [0], [0.9]

detector = MockAnomalyDetector()
env = FirewallEnvironment(detector)

# Training simulation
print("Training DQN Agent...")

# Define X_train and y_train with mock data for demonstration purposes
# Replace this with actual data loading logic
X_train = np.random.rand(100, 10)  # 100 samples, 10 features each
y_train = np.random.randint(0, 2, size=100)  # 100 labels (binary classification)

num_episodes = 100
episode_rewards = []
episode_losses = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    episode_loss = 0
    steps = 0
    
    # Run episode with random traffic samples
    for step in range(50):  # 50 steps per episode
        # Get random traffic sample
        sample_idx = random.randint(0, len(X_train) - 1)
        traffic_sample = X_train[sample_idx][-1]  # Get last timestep features
        true_label = y_train[sample_idx]
        
        # Agent chooses action
        action = agent.get_action(state)
        
        # Environment processes action
        next_state, reward, done = env.step(action, traffic_sample, true_label)
        
        # Store experience
        agent.remember(state, action, reward, next_state, done)
        
        # Update state and tracking
        state = next_state
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    # Train agent if enough experiences
    if len(agent.memory) > agent.batch_size:
        loss = agent.train(agent.memory)
        if loss is not None:
            episode_loss = loss
    
    episode_rewards.append(total_reward)
    episode_losses.append(episode_loss)
    
    if episode % 20 == 0:
        avg_reward = np.mean(episode_rewards[-20:])
        print(f"Episode {episode:3d} | Avg Reward: {avg_reward:7.2f} | Epsilon: {agent.epsilon:.3f}")

agent.training_rewards = episode_rewards
print(f"Training completed! Total episodes: {num_episodes}")

# Evaluate trained agent
print(f"\n=== Agent Evaluation ===")
test_episodes = 20
evaluation_rewards = []

# Define X_test and y_test with mock data for demonstration purposes
# Replace this with actual test data loading logic
X_test = np.random.rand(20, 10)  # 20 samples, 10 features each
y_test = np.random.randint(0, 2, size=20)  # 20 labels (binary classification)

agent.epsilon = 0.0  # Disable exploration for evaluation

for episode in range(test_episodes):
    state = env.reset()
    total_reward = 0
    
    for step in range(30):
        sample_idx = random.randint(0, len(X_test) - 1)
        traffic_sample = X_test[sample_idx][-1]
        true_label = y_test[sample_idx]
        
        action = agent.get_action(state)
        next_state, reward, done = env.step(action, traffic_sample, true_label)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    evaluation_rewards.append(total_reward)

avg_eval_reward = np.mean(evaluation_rewards)
print(f"Average evaluation reward: {avg_eval_reward:.2f}")
print(f"Standard deviation: {np.std(evaluation_rewards):.2f}")

# Show learned policy examples
print(f"\n=== Learned Policy Examples ===")
print("State → Action Mapping (showing Q-values for different network states):")

sample_states = [env.reset() for _ in range(5)]
for i, state in enumerate(sample_states):
    state_key = agent.get_state_key(state)
    if state_key in agent.q_table:
        q_values = agent.q_table[state_key]
        best_action = np.argmax(q_values)
        print(f"State {i+1}: Best Action = {agent.actions[best_action]} (Q-values: {q_values})")
    else:
        print(f"State {i+1}: Unseen state - would explore randomly")

# Performance summary
print(f"\n=== Performance Summary ===")
metrics = env.performance_metrics
total = metrics['total_packets']
if total > 0:
    print(f"Total packets processed: {total}")
    print(f"Threats blocked: {metrics['threats_blocked']}")
    print(f"False positives: {metrics['false_positives']} ({(metrics['false_positives']/total)*100:.1f}%)")
    print(f"False negatives: {metrics['false_negatives']} ({(metrics['false_negatives']/total)*100:.1f}%)")

print(f"\n✓ DQN Agent training and evaluation completed successfully")