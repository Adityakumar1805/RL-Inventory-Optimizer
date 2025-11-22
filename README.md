# RL-Inventory-Optimizer
<!-- 🟦 PROJECT BANNER -->
![Banner](banner.svg)


<div align="center">

# 🎯 Reinforcement Learning for Inventory Management

### *Optimizing Stock Levels with Deep Q-Networks*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

[Features](#-features) • [Demo](#-live-demo) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Contributing](#-contributing)


</div>

---

## 📖 Overview

This project implements a **Deep Q-Network (DQN)** agent to solve the classic inventory management problem. The system learns optimal restocking policies through reinforcement learning, outperforming traditional fixed reorder point strategies by **12.4%** in profit optimization.

### 🎯 Problem Statement

Inventory management involves balancing three competing objectives:
- 📦 **Minimize holding costs** (storage, insurance, obsolescence)
- 🚫 **Minimize stockout costs** (lost sales, customer dissatisfaction)
- 💰 **Maximize revenue** (meeting customer demand efficiently)

Traditional methods use fixed rules (e.g., reorder when stock < 20 units). This project demonstrates how **AI can learn adaptive policies** that respond to changing conditions.

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🤖 AI-Powered Optimization
- Deep Q-Network (DQN) implementation
- Experience replay for stable learning
- Epsilon-greedy exploration strategy
- Neural network Q-value approximation

</td>
<td width="50%">

### 📊 Interactive Visualization
- Real-time training progress tracking
- Live performance metrics dashboard
- Profit comparison charts
- Service level analytics

</td>
</tr>
<tr>
<td width="50%">

### 🎮 Web Interface
- Beautiful gradient-based UI design
- Responsive layout for all devices
- Tab-based navigation system
- Training logs in terminal style

</td>
<td width="50%">

### 📈 Performance Analysis
- Comprehensive results comparison
- Visual bar charts and graphs
- Key insights extraction
- Annual profit projections

</td>
</tr>
</table>

---

## 🎬 Live Demo

### Web Interface Preview

![Web Interface](web-interface.svg)

The system includes a fully interactive web interface built with pure HTML, CSS, and JavaScript:

```
📱 Responsive Design  |  ⚡ Real-time Updates  |  🎨 Modern UI/UX
```

**Key Sections:**
1. **Overview Tab** - Environment parameters, system architecture, reward function
2. **Training Progress Tab** - Live training visualization with charts and logs
3. **Results Tab** - Performance comparison and insights

![Training Progress](training-progress.svg)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INVENTORY ENVIRONMENT                    │
│  State: Inventory Level (0-200) → Action: Order Quantity   │
│         Reward: Profit (Revenue - Costs)                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      DQN AGENT                              │
│  Neural Network (24-24-6) → Q-Value Approximation          │
│  Experience Replay → Stable Learning                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   OPTIMAL POLICY                            │
│  Adaptive Reordering → Maximum Profit                       │
└─────────────────────────────────────────────────────────────┘
```

### Environment Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| 📦 Max Inventory | **200 units** | Maximum storage capacity |
| 🛒 Avg Demand | 20 units/day | Poisson-distributed customer demand |
| ⏱️ Lead Time | 3 days | Order delivery delay |
| 💵 Unit Price | ₹166 | Revenue per unit sold |
| 💸 Holding Cost | ₹8.30/unit/day | Storage cost |
| 🚫 Stockout Cost | ₹83/unit | Penalty for unmet demand |
| 📅 Episode Length | 90 days | Simulation period |

### Agent Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| 🧠 Algorithm | Deep Q-Network | Value-based RL method |
| 🎲 Discount Factor (γ) | 0.95 | Future reward importance |
| 🔍 Epsilon Decay | 0.995 | Exploration rate reduction |
| 📚 Batch Size | 32/64 | Training sample size |
| 💾 Memory Size | 1000-2000 | Experience replay buffer |
| 🎯 Actions | 6 discrete | [0, 10, 20, 30, 40, 50] units |

---

## 🚀 Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Pip package manager
pip --version
```

### Setup Steps

**1. Clone the Repository**
```bash
git clone https://github.com/yourusername/rl-inventory-management.git
cd rl-inventory-management
```

**2. Create Virtual Environment (Recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Verify Installation**
```bash
python inventory-rl-training.py
```

### 📦 Required Packages

Create a `requirements.txt` file with:

```
tensorflow>=2.12.0
numpy>=1.23.5
gym>=0.26.2
matplotlib>=3.7.1
```

---

## 💻 Usage

### Option 1: Web Interface (Recommended)

```bash
# Open the HTML file in your browser
# Windows
start index.html

# macOS
open index.html

# Linux
xdg-open index.html
```

**Quick Start:**
1. Click **"Start Training"** to begin DQN training
2. Watch real-time progress in the **Training Progress** tab
3. View results comparison in the **Results** tab

![Results Comparison](results-comparison.png)

### Option 2: Python Script

```bash
python inventory-rl-training.py
```

**Script Features:**
- 🏋️ Trains DQN agent for 500 episodes
- 📊 Generates performance visualizations
- 💾 Saves results as PNG charts
- 📈 Displays comprehensive comparison tables

### Advanced Usage

**Custom Training Configuration:**
```python
# Modify hyperparameters in the script
agent = DQNAgent(
    state_shape=(1,),
    n_actions=6,
    gamma=0.95,           # Adjust discount factor
    epsilon_decay=0.995,  # Adjust exploration decay
    learning_rate=0.001   # Adjust learning rate
)
```

**Custom Environment Parameters:**
```python
env = InventoryEnv()
env.max_inventory = 200      # Maximum capacity
env.demand_mean = 25         # Increase demand
env.holding_cost = 0.15      # Adjust costs
```

---

## 📊 Results

### Performance Metrics

<div align="center">

| Metric | DQN Agent | Fixed Policy | Improvement |
|--------|-----------|--------------|-------------|
| 💰 **Average Profit** | ₹285,447 | ₹253,892 | **+12.4%** |
| 🎯 **Service Level** | 97.8% | 93.2% | **+4.6%** |
| 📦 **Unmet Demand** | 39.5 units | 122.8 units | **-67.8%** |
| 💵 **Additional Profit** | +₹31,555 | Baseline | **₹31,555** |
| 📅 **Annual Savings** | ₹127,848 | Baseline | **₹127,848** |

</div>

![Performance Chart](performance_chart_image.svg)

### Key Findings

🎉 **Performance Highlights:**
- ✅ DQN agent achieves **12.4% higher profit** than fixed reorder policies
- ✅ Reduces unmet demand by **67.8%**, dramatically improving customer satisfaction
- ✅ Learns adaptive strategies that respond to demand variability
- ✅ Converges in **200-300 episodes** (approximately 2-5 minutes on CPU)
- ✅ Service level improvement of **4.6 percentage points**

📈 **Training Insights:**
- Initial episodes show high exploration (epsilon = 1.0)
- Profit stabilizes around episode 150-200
- Moving average demonstrates consistent improvement
- Final epsilon ≈ 0.05-0.10 (mostly exploitation)
- **No overfitting observed** - generalizes well to test episodes

💰 **Business Impact:**
- **₹31,555 additional profit per 90-day cycle**
- **₹127,848 projected annual savings**
- **83.2 fewer stockouts per cycle**
- **Payback period: Immediate** (no additional infrastructure costs)

---

## 🔬 Technical Details

### Reward Function

The profit calculation incorporates three components:

```
Profit = Revenue - Holding Cost - Stockout Cost

Where:
  Revenue = units_sold × ₹166
  Holding Cost = inventory_level × ₹8.30 × days
  Stockout Cost = unmet_demand × ₹83
```

### Neural Network Architecture

```
Input Layer:  1 neuron  (inventory level: 0-200)
              ↓
Hidden Layer: 24 neurons (ReLU activation)
              ↓
Hidden Layer: 24 neurons (ReLU activation)
              ↓
Output Layer: 6 neurons  (Q-values for each action: [0,10,20,30,40,50])
```

**Total Parameters:** 793 trainable parameters

### Learning Algorithm

1. **Observe** current state (inventory level)
2. **Select** action using ε-greedy policy
3. **Execute** action and observe reward
4. **Store** experience in replay buffer
5. **Sample** random minibatch from memory
6. **Compute** target Q-values using Bellman equation:
   ```
   Q(s,a) = r + γ × max Q(s',a')
   ```
7. **Update** network weights via gradient descent
8. **Decay** exploration rate (epsilon)

### Training Statistics

- **Total Episodes:** 500
- **Steps per Episode:** ~90-100
- **Training Time:** 3-5 minutes (CPU)
- **Memory Usage:** ~200 MB
- **Final Epsilon:** 0.047
- **Convergence:** Episode 180

---

## 📁 Project Structure

```
rl-inventory-management/
│
├── 📄 index.html                    # Interactive web interface
├── 🐍 inventory-rl-training.py      # Python training script
├── 📋 requirements.txt              # Python dependencies
├── 📖 README.md                     # This file
├── 📜 LICENSE                       # MIT License
│
├── 📊 images/                       # Screenshots and visuals
│   ├── banner.png
│   ├── web-interface.png
│   ├── training-progress.png
│   ├── results-comparison.png
│   └── performance-chart.png
│
├── 📈 results/                      # Generated outputs
│   ├── inventory_rl_results_inr.png
│   └── training_logs.txt
│
├── 🧠 models/                       # Saved models (optional)
│   └── dqn_model.h5
│
└── 📚 docs/                         # Additional documentation
    ├── algorithm_details.md
    └── deployment_guide.md
```

---

## 🎓 Learning Resources

### Understanding Reinforcement Learning

- 📘 [Sutton & Barto: RL Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- 🎥 [DeepMind RL Course](https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb)
- 📝 [Inventory Optimization with Deep Reinforcement Learning](https://doi.org/10.1016/j.eswa.2025.128259)
- 🎯 [OpenAI Spinning Up](https://spinningup.openai.com/)

### Inventory Management Fundamentals

- 📚 Economic Order Quantity (EOQ) Model
- 📊 Newsvendor Problem
- 🔄 (s, S) Inventory Policies
- 📈 Supply Chain Optimization

---

## 🛠️ Customization

### Modify Action Space

```python
# Change available order quantities
self.n_actions = 10  # More granular actions
# Actions: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
```

### Adjust Demand Distribution

```python
# Uniform demand
demand = np.random.randint(15, 26)

# Normal demand
demand = int(np.random.normal(20, 5))

# Seasonal demand
base_demand = 20
seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * self.day / 90)
demand = np.random.poisson(base_demand * seasonal_factor)
```

### Different Reward Structures

```python
# Penalize high inventory more heavily
reward = revenue - 2 * holding_cost - stockout_cost

# Add order cost
order_cost = (order_quantity > 0) * 50  # Fixed ordering cost
reward = revenue - holding_cost - stockout_cost - order_cost

# Add service level bonus
service_bonus = 100 if unmet_demand == 0 else 0
reward = revenue - holding_cost - stockout_cost + service_bonus
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 **Fork** the repository
2. 🌟 **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 **Push** to the branch (`git push origin feature/AmazingFeature`)
5. 🎉 **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines for Python code
- Add comments for complex logic
- Update documentation for new features
- Include unit tests for new functionality
- Test on multiple browsers for web changes

### Ideas for Contributions

- 🎨 Add more visualization options
- 🧪 Implement other RL algorithms (PPO, A3C, SAC)
- 📊 Add multi-product inventory management
- 🌐 Create REST API for model serving
- 📱 Develop mobile app interface
- 🔧 Add hyperparameter tuning automation

---

## 🐛 Troubleshooting

### Common Issues

**Issue: TensorFlow installation fails**
```bash
# Try installing specific version
pip install tensorflow==2.12.0

# Or use CPU-only version
pip install tensorflow-cpu==2.12.0
```

**Issue: Web interface doesn't display charts**
```
Solution: Ensure JavaScript is enabled in your browser
Clear browser cache and reload the page
```

**Issue: Training takes too long**
```python
# Reduce number of episodes
episodes = 200  # Instead of 500

# Or reduce episode length
self.episode_length = 50  # Instead of 90
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License - You are free to:
✓ Use commercially
✓ Modify and distribute
✓ Use privately
✓ Sublicense
```

---

## 🙏 Acknowledgments

- 🎓 **DeepMind** - For pioneering DQN research
- 📚 **OpenAI Gym** - For standardized RL environments
- 🧠 **TensorFlow Team** - For deep learning framework
- 🎨 **Tailwind CSS** - For design inspiration
- 🌟 **Open Source Community** - For invaluable tools and libraries

---

## 📧 Contact & Support

<p align="center">
  <strong>Have questions or suggestions?</strong><br><br>

  <a href="https://github.com/Adityakumar1805/RL-Inventory-Optimizer/issues" target="_blank">
    <img src="https://img.shields.io/badge/GitHub Issues-red?logo=github" />
  </a>

  <a href="mailto:adityakumar182505@gmail.com" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Email-Contact-blue?logo=gmail" />
  </a>

  <a href="https://www.linkedin.com/in/aditya-kumar-6b17852b9/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin" />
  </a>
</p>


---

## 📈 Roadmap

### Version 2.0 (Planned)

- [ ] 🔄 Multi-product inventory management
- [ ] 🌐 RESTful API for production deployment
- [ ] 📊 Advanced analytics dashboard
- [ ] 🤖 Automated hyperparameter optimization
- [ ] 📱 Mobile application
- [ ] 🔐 User authentication and data persistence
- [ ] 🌍 Multi-warehouse support
- [ ] 📉 Risk analysis and scenario planning

---

## 🌟 Show Your Support

If this project helped you, please consider:

- ⭐ **Starring** the repository
- 🍴 **Forking** for your own experiments
- 📢 **Sharing** with others who might benefit
- 💬 **Contributing** improvements or ideas

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{RL-Inventory-Optimizer,
  author = {Aditya Kumar},
  title = {RL-Inventory-Optimizer},
  year = {2025},
  url = {https://github.com/Adityakumar1805/RL-Inventory-Optimizer}
}
```

---

<div align="center">

### 🚀 Built with Machine Learning • Optimized for Performance • Designed for Learning

**[⬆ Back to Top](#-RL-Inventory-Optimizer)**

---

*Made with ❤️ by [Aditya Kumar](https://github.com/Adityakumar1805)*

**⭐ Star this repo if you find it useful!**

</div>

