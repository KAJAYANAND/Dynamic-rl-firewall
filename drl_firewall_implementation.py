# Dynamic Reinforcement Learning Firewall - Production Implementation

This module contains the core classes used in deployment for the Dynamic Reinforcement Learning Firewall (DRL-Firewall). 

---

## Technical Report

### Executive Summary

The implementation demonstrates a novel **Dynamic Reinforcement Learning Firewall (DRL-Firewall)** that combines deep learning-based anomaly detection with reinforcement learning-based policy optimization to deliver adaptive, intelligent, and self-improving network security.

---

## System Architecture

### Core Components

1. **LSTM-CNN Hybrid Anomaly Detector**
   - Processes sequential network traffic data
   - Combines temporal (LSTM) and spatial (CNN) feature extraction
   - Detects suspicious or abnormal network behaviors

2. **DQN Firewall Agent**
   - Learns optimal blocking/allowing actions using Deep Q-Networks
   - Balances security level and network performance
   - Continuously improves through feedback and rewards

3. **Firewall Environment**
   - Simulates real-time network traffic
   - Provides dynamic feedback to the RL agent
   - Supports adaptive policy updates for evolving threats

---

## Deployment
The trained DRL-Firewall can be integrated into an **SDN Controller** or **Edge Gateway**, enabling:
- Real-time anomaly detection and mitigation
- Dynamic policy adaptation
- Reduced false positives and improved throughput

---

## Source Structure

- **`LSTMCNNAnomalyDetector`**: A hybrid deep learning model responsible for identifying anomalies in network traffic. It combines a Long Short-Term Memory (LSTM) network to capture temporal patterns and a Convolutional Neural Network (CNN) to extract spatial features.

- **`DQNFirewallAgent`**: The core reinforcement learning agent that learns and applies firewall policies. It uses a Deep Q-Network (DQN) to decide whether to block or allow traffic based on the anomaly scores from the detector and the state of the network.

- **`FirewallEnvironment`**: A simulated environment that mimics real-world network traffic and conditions. It provides the agent with state information, receives actions (block/allow), and returns rewards based on the impact of those actions on security and network performance.