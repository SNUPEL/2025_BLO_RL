# 2025_BLO_RL

|                 Developer                |                 
| :--------------------------------------: | 
| [Seung Woo Han](https://github.com/SeungwooHH11) |
|         🧑‍💻 AI-Development               |       
<br>

## Project Overview
- **Project**
    - Integrated shipyard block logistic optimization
- **Superviser**
    - Prof. Jong Hun Woo (SNU, KOREA)
- **Data provider**
    - HD Korea Shipbuilding & Offshore Engineering (KSOE)

<br>

## Project Introduction
We develop a integrated shipyard block logistic optimization algorithm for shipyards 
<br>
Block logistic is composed of block transportation scheduling and storage optimization in block storage yard <br>
Our project focued on **integrated simulation of two main problem and joint optimization algorithm**


<img src="BRP_figure/BLO_problem.png"/>
<br>


## Main Function

### 1️⃣ Overall framework
#### 1.1 Algorithm overview
<img src="BRP_figure/BLO_framework.png"/>

- Retrieval and placement are considered independent decisions. <br>
- The integrated optimization is achieved through the scheduling algorithm.<br>

<br>

### 2️⃣ Markov decision process

#### 2.1 State
- State composed of unscheduled blocks and transporters
    - **edge attributed graph**: compact and effective representation of block transportation statue
        - nodes representing location which contain current transporter information
        - edges representing blocks with origin and destination by disjunctive edge
    - **Crystal graph convolutional neural network**: graph neural network that suitable for encoding edge attributed graph

#### 2.2 Action
- Assigining next transportation block for earilest finishing tranporter
    - **candidate blocks**
        - weight capacity constraint
        - ready time constraint 
    - **candidate transporter**
        - rule based agent selection by earliest finishing tranporter

#### 2.3 Reward
- minimization of the total coss
- a sum of three cost-related rewards
    - **Empty travel time**: the penalty cost for the delay in the delivery of vessels
    - **Tardiness**: the cost of moving the vessels
    - **Block relocation**: the additional processing cost

<br>

### 3️⃣ DES-based learning environment
- DES model of the post-stage outfitting process in shipyards
- state transition that takes the action of the agent as the input and calculates the next state and reward.

<br>

### 4️⃣ Scheduling agent with PPO algorithm
#### 4.1 Network Structure
<img src="BRP_figure/TP_network_structure.png"/>


- **Representation module**
    - Two types of latent representation are extracted from the heterogeneous graphs and auxiliary matrix, respectively
    - For heterogeneous graphs, the embedding vectors of nodes are generated using the relational information between nodes
    - For an auxiliary matrix, the embedding vectors for combinations of quay-walls and vessels are generated using the MLP layers 
- **Aggregation module**
    - Input vectors for the output model are generated based on the embedding vectors from the representation module
- **Output module**
    - The actor layers calculate the probability distribution over actions $\pi_{\theta} (\cdot|s_t)$
    - The critic layers calculate a approximate state-value function $V_{\pi_{\theta}} (s_t)$, respectively

#### 4.2 Reinforcement Learning Algorithm
- **PPO(proximal policy optimization)**
    - Policy-based reinforcement learning algorithm

