from mesa import Agent
import numpy as np

class Woman(Agent):
    
    def __init__(self, unique_id, model, alpha=1, beta=1, block_threshold=0.3, biased_activation=False):
        super().__init__(model)
        self.unique_id = unique_id  
        # Initialize grb with Beta distribution (default uniform)
        self.grb = np.random.beta(alpha, beta) 
        self.received_info = {}  # Store grb info received this step: {agent_id: grb}
        self.block_risk = {}       # {agent_id: blocking risk}
        self.blocked_agents = set()  # permanently blocked agents
        self.block_threshold = block_threshold  # Blocking threshold, user-defined
        self.biased_activation = biased_activation  # Flag to enable biased activation

    def step(self):
        #Clear info storage at the start of each step
        self.received_info.clear()

    def activate_and_send_info(self):
        # Only activated agents send info to neighbors + random others
        neighbors = list(self.model.network.neighbors(self.unique_id))
        
        # Select some random agents that are NOT neighbors and not self
        all_agents = set(self.model.network.nodes())
        non_neighbors = list(all_agents - set(neighbors) - {self.unique_id})
        
        # Fixed proportion of random connections for all agents
        random_proportion = 0.2  # Default 20% of random others
        if self.biased_activation:
            if self.grb > 0.6:
                random_proportion = 0.5
                
        # Number of random others to send info to
        num_random = int(len(non_neighbors) * random_proportion)
        random_others = self.random.sample(non_neighbors, num_random) if num_random > 0 else []
        
        receivers = neighbors + random_others
        
        # Only send information to receivers (no immediate receiving back)
        for r_id in receivers:
            agent = self.model.get_agent_by_id(r_id)
            if r_id not in self.blocked_agents:  # Only check if we haven't blocked them
                agent.receive_info(self.unique_id, self.grb)

    def receive_info(self, sender_id, sender_grb):
        """Store the received belief from another agent"""
        self.received_info[sender_id] = sender_grb

    def process_received_info(self):
        """Process any information received from other agents"""
        if not self.received_info:
            return

        # Filter out information from blocked agents
        valid_info = {
            sender_id: grb 
            for sender_id, grb in self.received_info.items() 
            if sender_id not in self.blocked_agents
        }

        if not valid_info:  # If no valid information after filtering
            return

        # Update blocking risks based on received information
        for sender_id, sender_grb in valid_info.items():
            diff = abs(self.grb - sender_grb)
            if diff > self.block_threshold:
                current_risk = self.block_risk.get(sender_id, 0)
                self.block_risk[sender_id] = current_risk + diff 
                
                if self.block_risk[sender_id] >= 1.0:
                    self.blocked_agents.add(sender_id)
                    self.block_risk.pop(sender_id, None)

        # Update own belief based on valid (non-blocked) information
        mean_received_grb = np.mean(list(valid_info.values()))
        learning_rate = 0.3
        self.grb += learning_rate * (mean_received_grb - self.grb)
        self.grb = max(0, min(1, self.grb))  # Ensure GRB stays within [0,1]

        # Clear received info for next step
        self.received_info.clear()
