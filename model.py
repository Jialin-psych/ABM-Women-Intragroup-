import numpy as np
from mesa import Model
from mesa.space import NetworkGrid
from agents import Woman
from networkx import Graph
from mesa.datacollection import DataCollector

class GenderBeliefModel(Model):

    def __init__(self, num_agents = 100, alpha=1, beta=1, block_threshold=0.3, biased_activation=False, activation_fraction=0.3, similarity_threshold=0.2):
        super().__init__()
        self.num_agents = num_agents
        self.alpha = alpha
        self.beta = beta
        self.block_threshold = block_threshold
        self.biased_activation = biased_activation
        self.activation_fraction = activation_fraction
        self.similarity_threshold = similarity_threshold

        # Create a network for the agents
        self.network = Graph()
        self.grid = NetworkGrid(self.network)
        self._agents = {}  # Store agents by ID

        # Create agents and add them to the model and network
        for i in range(self.num_agents):
            agent = Woman(unique_id=i, model=self, alpha=self.alpha, beta=self.beta,
                          block_threshold=self.block_threshold, biased_activation=self.biased_activation)
            if agent is None:
                print(f"Agent creation failed for id {i}")
            self._agents[i] = agent
            self.network.add_node(i, agent=[])   # add node by agent ID
            self.grid.place_agent(agent, i)  # place agent on node i

        # Add edges between agents based on similarity
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
        for agent_id in range(self.num_agents):
            for other_id in range(agent_id + 1, self.num_agents):
                agent_grb = self._agents[agent_id].grb
                other_grb = self._agents[other_id].grb
                # Create an edge if the beliefs are similar enough and no existing edge
                if abs(agent_grb - other_grb) < self.similarity_threshold:
                    if not self.network.has_edge(agent_id, other_id):
                        self.network.add_edge(agent_id, other_id)

    def step(self):
        self.datacollector.collect(self)

        agent_ids = list(self._agents.keys())
        n_activate = int(len(agent_ids) * self.activation_fraction)
        activated_ids = self.random.sample(agent_ids, n_activate)

        for agent_id in activated_ids:
            agent = self._agents[agent_id]
            agent.activate_and_send_info()  # active senders share info

        for agent in self._agents.values():
            agent.process_received_info()  # all update beliefs based on info


    def compute_global_grb(self):
        agent_grbs = [agent.grb for agent in self._agents.values() if agent is not None]
        return np.mean(agent_grbs) if agent_grbs else 0


    def get_agent_data(self):
        """
        Collect data about agents for visualization.
        """
        return self.datacollector.get_agent_vars_dataframe()

    def get_global_data(self):
        """
        Get the global data (average GRB).
        """
        return self.datacollector.get_model_vars_dataframe()
