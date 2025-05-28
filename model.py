import numpy as np
from mesa import Model
from mesa.space import NetworkGrid
from agents import Woman
from networkx import Graph
from mesa.datacollection import DataCollector

class GenderBeliefModel(Model):
    def __init__(self, num_agents=100, alpha=1, beta=1, block_threshold=0.3, 
                 biased_activation=False, activation_fraction=0.3, similarity_threshold=0.2, seed=42):
        # Set random seed for reproducibility
        super().__init__(seed=seed)
        np.random.seed(seed)
        
        self.num_agents = num_agents
        self.alpha = alpha
        self.beta = beta
        self.block_threshold = block_threshold
        self.biased_activation = biased_activation
        self.activation_fraction = activation_fraction
        self.similarity_threshold = similarity_threshold
        
        # Create network and grid
        self.network = Graph()
        self.grid = NetworkGrid(self.network)
        
        # Store agents (using custom attribute name to avoid conflicts with Mesa's default agent list)
        self.agent_list = []
        
        # Create agents
        for i in range(self.num_agents):
            # Create the agent
            agent = Woman(unique_id=i, model=self, alpha=self.alpha, beta=self.beta,
                         block_threshold=self.block_threshold, biased_activation=self.biased_activation)
            
            # Add agent to model's custom agent list
            self.agent_list.append(agent)
            
            # Add node to network 
            self.network.add_node(i)
            self.network.nodes[i]["agent"] = []  # Initialize as empty list for Mesa
            
            # Place agent on the network grid 
            self.grid.place_agent(agent, i)
        
        # Connect agents based on similarity
        self.connect_similar_agents()
        
        # Set up data collection
        self.datacollector = DataCollector(
            agent_reporters={"grb": "grb"},
            model_reporters={"Average_GRB": self.compute_global_grb}
        )

    def connect_similar_agents(self):
        """
        Connect agents based on similarity of their beliefs (grb).
        Agents with more similar beliefs (closer grb values) are more likely to be connected.
        """
        # Use model's custom agent list
        agents_list = self.agent_list
        
        for i, agent in enumerate(agents_list):
            for j in range(i + 1, len(agents_list)):
                other_agent = agents_list[j]
                
                # Calculate similarity based on GRB values
                if abs(agent.grb - other_agent.grb) < self.similarity_threshold:
                    if not self.network.has_edge(agent.unique_id, other_agent.unique_id):
                        self.network.add_edge(agent.unique_id, other_agent.unique_id)

    def step(self):
        self.datacollector.collect(self)
        
        # Get all agents from custom list
        all_agents = self.agent_list
        
        # Activate a fraction of agents
        n_activate = int(len(all_agents) * self.activation_fraction)
        activated_agents = self.random.sample(all_agents, n_activate)
        
        # Active agents send information
        for agent in activated_agents:
            agent.activate_and_send_info()
        
        # All agents process received information
        for agent in all_agents:
            agent.process_received_info()

    def compute_global_grb(self):
        """Compute the average GRB across all agents."""
        agent_grbs = [agent.grb for agent in self.agent_list]
        return np.mean(agent_grbs) if agent_grbs else 0

    def get_agent_data(self):
        """Collect data about agents for visualization."""
        return self.datacollector.get_agent_vars_dataframe()

    def get_global_data(self):
        """Get the global data (average GRB)."""
        return self.datacollector.get_model_vars_dataframe()
    
    # Helper method to access agents by ID if needed
    def get_agent_by_id(self, agent_id):
        """Get agent by unique_id."""
        for agent in self.agent_list:
            if agent.unique_id == agent_id:
                return agent
        return None