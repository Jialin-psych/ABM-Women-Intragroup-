import networkx as nx
from mesa import Model
from mesa.space import NetworkGrid
from mesa.time import RandomActivation
from random import randint
from agents import Woman       

class WomenIntragroupModel(Model):

    def __init__(self, num_agents: int,
                 alpha: float = 20.0,      # homophily steepness
                 seed: int | None = None):
        super().__init__(seed=seed)

        self.num_agents = num_agents
        self.alpha      = alpha            # smaller = stricter homophily
        self.schedule   = RandomActivation(self)

        # -------- 1. Create agents & remember their GRB ---------------------
        grb_values: list[int] = []
        for uid in range(num_agents):
            grb = self.random.randint(0, 100)       # single RNG stream
            agent = Woman(uid, self, grb)
            self.schedule.add(agent)
            grb_values.append(grb)

        # -------- 2. Build the homophily-based graph ------------------------
        G = nx.Graph()
        G.add_nodes_from(range(num_agents))

        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # edge probability falls off with GRB distance
                p = self._edge_prob(grb_values[i], grb_values[j])
                if self.random.random() < p:
                    G.add_edge(i, j)

        # -------- 3. Wrap in Mesa’s NetworkGrid & place agents --------------
        self.network = NetworkGrid(G)
        for node_id, agent in enumerate(self.schedule.agents):
            self.network.place_agent(agent, node_id)

        self.running = True   

    def _edge_prob(self, grb_i: int, grb_j: int) -> float:
        """Exponential homophily kernel."""
        delta = abs(grb_i - grb_j)
        return pow(2.718281828459045, -delta / self.alpha)


    def step(self):
        self.schedule.step()
