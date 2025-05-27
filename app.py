import solara
from model import GenderBeliefModel
from mesa.visualization import (
    SolaraViz,
    make_space_component,
    make_plot_component,
)


# Agent portrayal function with gradual blue and gray for blocked agents
def agent_portrayal(agent):
    blue_value = agent.grb  # a float between 0 and 1
    # Create normalized RGB tuple (R,G,B) with B scaled by blue_value
    color = (0, 0, blue_value) 
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
}

# Instantiate model
grb_model = GenderBeliefModel()
# Function returning average GRB for plotting
def average_grb_measure(model):
    return model.compute_global_grb()

# Create visualization components
space_graph = make_space_component(
    agent_portrayal,
    draw_grid=False,
)

average_grb_plot = make_plot_component(
    measure=average_grb_measure,
)

# Create SolaraViz page combining components and exposing parameters
page = SolaraViz(
    grb_model,
    components=[space_graph, average_grb_plot],
    model_params=model_params,
    name="Gender Role Belief Model",
)

# Return the page for solara to render
page
