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
        #Activated agents send info to neighbors + random others, excluding blocked agents.
        neighbors = list(self.model.network.neighbors(self.unique_id))
        
        # Select some random agents that are NOT neighbors and not self
        all_agents = set(self.model.network.nodes())
        non_neighbors = list(all_agents - set(neighbors) - {self.unique_id})
        
        # Adjust the proportion of random agents based on belief (if biased)
        random_proportion = 0.2  # Default 20% of random others
        if self.biased_activation:
        # Increase random proportion for agents with higher grb (self.grb > 0.6)
            if self.grb > 0.6:
            # If agent's belief is high, increase the random proportion to 50%
                random_proportion = 0.5
                
        # Number of random others to send info to
        num_random = int(len(non_neighbors) * random_proportion)
        random_others = self.random.sample(non_neighbors, num_random) if num_random > 0 else []
        
        receivers = neighbors + random_others
        
        # Send own grb to receivers
        for r_id in receivers:
            agent = self.model._agents[r_id]
            if r_id not in self.blocked_agents and self.unique_id not in agent.blocked_agents:
                agent.receive_info(self.unique_id, self.grb)
        
        # Receive back info from receivers (if not blocked)
        for r_id in receivers:
            agent = self.model._agents[r_id]
            if r_id not in self.blocked_agents and self.unique_id not in agent.blocked_agents:
                self.receive_info(r_id, agent.grb)

    def receive_info(self, sender_id, sender_grb):
        """Store the received belief from another agent"""
        self.received_info[sender_id] = sender_grb

    def process_received_info(self):
        """Process the information received and update belief and blocking risk"""
        if not self.received_info:
            return

        # Check edges with agents where grb difference is significant based on user-defined threshold
        for sender_id, sender_grb in self.received_info.items():
            diff = abs(self.grb - sender_grb)
            if diff > self.block_threshold:  # Use user-defined threshold for blocking
                # Increase blocking risk
                current_risk = self.block_risk.get(sender_id, 0)
                self.block_risk[sender_id] = current_risk + diff * 0.5  # Incremental risk
                
                # Threshold to block permanently
                if self.block_risk[sender_id] >= 1.0:
                    self.blocked_agents.add(sender_id)
                    # Once blocked, clear block risk to avoid confusion
                    self.block_risk.pop(sender_id, None)

        # Calculate mean difference to update own grb
        diffs = [abs(self.grb - val) for val in self.received_info.values()]
        mean_diff = np.mean(diffs)

        # Update grb towards the mean of received grbs with some learning rate
        mean_received_grb = np.mean(list(self.received_info.values()))
        learning_rate = 0.1  # Adjust this to control how quickly grb changes
        self.grb += learning_rate * mean_diff * (mean_received_grb - self.grb)
        self.grb = max(0, min(1, self.grb))
