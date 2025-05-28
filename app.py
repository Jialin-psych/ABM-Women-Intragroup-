import solara
from model import GenderBeliefModel
from mesa.visualization import (
    SolaraViz,
    make_space_component,
    make_plot_component,
)

def agent_portrayal(agent):
    # Create a color gradient from red (low GRB) to blue (high GRB)
    red = 1 - agent.grb  # red decreases as GRB increases
    blue = agent.grb     # blue increases as GRB increases
    color = (red, 0, blue)  # RGB tuple
    return {
        "color": color,
        "marker": 'o',
        "size": 20,
        "label": f"GRB: {agent.grb:.2f}",
    }
    
# Model parameters exposed as UI controls
model_params = {
    "num_agents": {
        "type": "SliderInt",
        "value": 100,
        "label": "Number of Agents",
        "min": 10,
        "max": 500,
        "step": 10,
    },
    "alpha": {
        "type": "SliderFloat",
        "value": 1,
        "label": "Beta Distribution Alpha",
        "min": 0.1,
        "max": 5,
        "step": 0.1,
    },
    "beta": {
        "type": "SliderFloat",
        "value": 1,
        "label": "Beta Distribution Beta",
        "min": 0.1,
        "max": 5,
        "step": 0.1,
    },
    "block_threshold": {
        "type": "SliderFloat",
        "value": 0.3,
        "label": "Blocking Threshold",
        "min": 0,
        "max": 1,
        "step": 0.05,
    },
    "biased_activation": {
        "type": "Checkbox",
        "value": True,
        "label": "Biased Activation",
    },
    "activation_fraction": {
        "type": "SliderFloat",
        "value": 0.3,
        "label": "Activation Fraction",
        "min": 0,
        "max": 1,
        "step": 0.05,
    },
    "similarity_threshold": {
        "type": "SliderFloat",
        "value": 0.2,
        "label": "Similarity Threshold",
        "min": 0,
        "max": 1,
        "step": 0.05,
    },
    "seed": {
        "type": "SliderInt",
        "value": 42,
        "label": "Random Seed",
        "min": 0,
        "max": 100,
        "step": 1,
    },
}

grb_model =GenderBeliefModel()

# Create visualization components
space_graph = make_space_component(
    agent_portrayal,
    draw_grid=False,
)

# Create plot component for average GRB over time
average_grb_plot = make_plot_component(
    "Average_GRB"  
)

# Create SolaraViz page combining components and exposing parameters
page = SolaraViz(
    grb_model,  
    components=[space_graph, average_grb_plot],
    model_params=model_params,
    name="Gender Role Belief Model",
)