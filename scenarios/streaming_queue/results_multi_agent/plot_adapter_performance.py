#!/usr/bin/env python3
"""
Script to plot adapter performance vs agent count
X-axis: agent count (4, 8, 16, 32)
Y-axis: average adapter time
Different protocols use different colored curves
"""

import json
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Configuration
agent_counts = [4, 8, 16, 32]
protocols = ['a2a', 'acp', 'agora', 'anp']
protocol_colors = {
    'a2a': '#FF6B6B',      # Red
    'acp': '#4ECDC4',      # Cyan
    'agora': '#45B7D1',    # Blue
    'anp': '#FFA07A'       # Orange
}
protocol_names = {
    'a2a': 'A2A',
    'acp': 'ACP',
    'agora': 'AGORA',
    'anp': 'ANP'
}

base_dir = Path('/root/AgentProtocols/scenarios/streaming_queue')

# Store data
data = {protocol: [] for protocol in protocols}

# Read data
for agent_count in agent_counts:
    results_dir = base_dir / f'results_{agent_count}agents'
    
    for protocol in protocols:
        json_file = results_dir / f'qa_results_{protocol}.json'
        
        try:
            with open(json_file, 'r') as f:
                result_data = json.load(f)
                
            # Extract average_adapter_time
            avg_adapter_time = result_data['timing_breakdown']['average_adapter_time']
            data[protocol].append(avg_adapter_time * 1000)  # Convert to milliseconds
            
            print(f"Agent {agent_count}, Protocol {protocol}: {avg_adapter_time * 1000:.6f} ms")
            
        except FileNotFoundError:
            print(f"Warning: File not found {json_file}")
            data[protocol].append(None)
        except KeyError as e:
            print(f"Warning: Data format error {json_file}: {e}")
            data[protocol].append(None)

# Plot the chart
plt.figure(figsize=(12, 8))

for protocol in protocols:
    # Filter out None values
    valid_data = [(x, y) for x, y in zip(agent_counts, data[protocol]) if y is not None]
    
    if valid_data:
        x_vals, y_vals = zip(*valid_data)
        plt.plot(x_vals, y_vals, 
                marker='o', 
                linewidth=2.5, 
                markersize=8,
                label=protocol_names[protocol],
                color=protocol_colors[protocol])

plt.xlabel('Number of Agents', fontsize=14, fontweight='bold')
plt.ylabel('Average Adapter Time (ms)', fontsize=14, fontweight='bold')
plt.title('Adapter Performance vs Agent Count for Different Protocols', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='best', frameon=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(agent_counts, fontsize=12)
plt.yticks(fontsize=12)

# Add data point labels
for protocol in protocols:
    valid_data = [(x, y) for x, y in zip(agent_counts, data[protocol]) if y is not None]
    if valid_data:
        for x, y in valid_data:
            plt.annotate(f'{y:.4f}', 
                        xy=(x, y), 
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=9,
                        alpha=0.7)

plt.tight_layout()

# Save the chart
output_file = base_dir / 'adapter_performance_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nChart saved to: {output_file}")

# Also save as PDF format
output_pdf = base_dir / 'adapter_performance_plot.pdf'
plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
print(f"Chart saved to: {output_pdf}")

plt.show()

# Print summary statistics
print("\n=== Data Summary ===")
for protocol in protocols:
    print(f"\n{protocol_names[protocol]}:")
    for agent_count, time in zip(agent_counts, data[protocol]):
        if time is not None:
            print(f"  {agent_count} agents: {time:.6f} ms")
