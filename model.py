from mesa import Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa import Agent
from random import randint
from woman import Woman  # Import the Woman agent class

class GenderEqualityModel(Model):
    """
    A model simulating the intragroup dynamics of women based on gender role beliefs and submission levels.
    The model uses an agent-based approach where agents (women) are placed in a network and can interact.
    """

    def __init__(self, num_agents, network):
        """
        Initializes the model with agents and a social network.
        
        Args:
            num_agents (int): Number of agents in the model.
            network (NetworkGrid): The social network where agents are connected.
        """
        self.num_agents = num_agents
        self.network = network
        
        # Create a scheduler to manage the agents
        self.schedule = RandomActivation(self)
        
        # Create agents with random Gender Role Beliefs (GRB) between 0 and 100
        for i in range(self.num_agents):
            grb = randint(0, 100)  # Assign random GRB to each agent
            a = Woman(i, self, grb)  # Create a Woman agent with the generated GRB
            self.schedule.add(a)  # Add the agent to the scheduler
            
            # Add agents to the network (fully connected for simplicity)
            if i > 0:
                self.network.add_edge(i, i - 1)  # Connect to the previous agent (can be customized)
        
        self.running = True

    def step(self):
        """
        Advances the model by one step.
        In each step, each agent interacts with its neighbors based on submission levels and GRBs.
        """
        self.schedule.step()  # Perform a step for each agent in the model
