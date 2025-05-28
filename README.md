# Gender Role Belief Model

An agent-based model simulating the dynamics of gender role beliefs in social networks, implemented using Mesa and Solara for visualization.

## Overview

This model simulates how gender role beliefs (GRB) evolve in a social network where agents can:
- Share their beliefs with neighbors and random others
- Block interactions with agents holding significantly different beliefs
- Update their beliefs based on received information
- Form connections based on belief similarity

## Model Parameters

- **Number of Agents**: Total number of agents in the network (10-500)
- **Beta Distribution Parameters**:
  - **Alpha**: Shape parameter for initial GRB distribution (0.1-5)
  - **Beta**: Shape parameter for initial GRB distribution (0.1-5)
- **Blocking Threshold**: Minimum belief difference that triggers blocking risk (0-1)
- **Biased Activation**: When enabled, agents with high GRB (>0.6) reach out to more random others
- **Activation Fraction**: Proportion of agents activated each step (0-1)
- **Similarity Threshold**: Maximum belief difference for initial network connections (0-1)
- **Random Seed**: For reproducible results (0-100)

## Visualization

The model provides two main visualizations:
1. **Network Graph**:
   - Nodes represent agents
   - Colors indicate GRB values (Red = low GRB, Blue = high GRB)
   - Edges show connections between agents
   - Hover over nodes to see exact GRB values

2. **Average GRB Plot**:
   - Shows the evolution of average GRB over time
   - Helps track global trends in belief changes

## Key Mechanisms

1. **Belief Sharing**:
   - Activated agents share their GRB with:
     - Network neighbors
     - Random subset of non-neighbors (20% default, 50% for high GRB if biased activation is enabled)

2. **Blocking Mechanism**:
   - Agents track "blocking risk" for others with significantly different beliefs
   - Risk accumulates based on belief differences
   - Once blocking threshold is reached, the agent permanently blocks the sender

3. **Belief Updates**:
   - Agents update their beliefs based on valid (non-blocked) received information
   - Updates are proportional to the difference between current and received beliefs
   - Learning rate controls the speed of belief changes

## Installation
To install the dependencies use pip and the requirements.txt in this directory. e.g.

```
$ pip install -r requirements.txt
```

To run the model interactively:

```
$ solara run app.py
```

## Files Structure

- `app.py`: Main application file with visualization setup
- `model.py`: Core model implementation
- `agents.py`: Agent class definition and behavior
- `README.md`: This documentation

