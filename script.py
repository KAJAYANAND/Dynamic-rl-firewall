import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Create a simplified implementation of Dynamic Reinforcement Learning Firewall components
print("=== Dynamic Reinforcement Learning Firewall Implementation ===\n")

# 1. Generate synthetic network traffic data
np.random.seed(42)

def generate_network_traffic(n_samples=1000):
    """Generate synthetic network traffic data with features"""
    data = {
        'packet_size': np.random.normal(1500, 500, n_samples),
        'connection_duration': np.random.exponential(2.0, n_samples),
        'source_port': np.random.randint(1024, 65536, n_samples),
        'dest_port': np.random.choice([80, 443, 22, 21, 25, 53], n_samples, p=[0.4, 0.3, 0.1, 0.05, 0.05, 0.1]),
        'protocol': np.random.choice([6, 17, 1], n_samples, p=[0.7, 0.25, 0.05]),  # TCP, UDP, ICMP
        'bytes_sent': np.random.lognormal(8, 2, n_samples),
        'bytes_received': np.random.lognormal(7, 2, n_samples),
        'flow_duration': np.random.exponential(5.0, n_samples)
    }
    
    # Generate labels (0=normal, 1=attack)
    # Create different attack patterns
    attack_indices = np.random.choice(n_samples, size=int(0.15 * n_samples), replace=False)
    labels = np.zeros(n_samples)
    labels[attack_indices] = 1
    
    # Modify attack traffic characteristics
    for idx in attack_indices:
        attack_type = np.random.choice(['ddos', 'port_scan', 'data_exfil'])
        if attack_type == 'ddos':
            data['packet_size'][idx] = np.random.normal(64, 10)  # Small packets
            data['bytes_sent'][idx] = np.random.normal(100, 20)
        elif attack_type == 'port_scan':
            data['connection_duration'][idx] = np.random.normal(0.1, 0.02)  # Very short
            data['dest_port'][idx] = np.random.randint(1, 1024)  # System ports
        elif attack_type == 'data_exfil':
            data['bytes_sent'][idx] = np.random.normal(10000, 2000)  # Large uploads
    
    df = pd.DataFrame(data)
    df['label'] = labels
    return df

# Generate traffic data
traffic_data = generate_network_traffic(2000)
print("Generated network traffic data:")
print(f"Total samples: {len(traffic_data)}")
print(f"Normal traffic: {sum(traffic_data['label'] == 0)}")
print(f"Attack traffic: {sum(traffic_data['label'] == 1)}")
print(f"Attack percentage: {(sum(traffic_data['label'] == 1) / len(traffic_data)) * 100:.1f}%\n")

# Show sample data
print("Sample network traffic features:")
print(traffic_data.head())
print("\nTraffic statistics:")
print(traffic_data.describe())

# Save traffic data for analysis
traffic_data.to_csv('network_traffic_data.csv', index=False)
print(f"\n✓ Network traffic data saved to 'network_traffic_data.csv'")