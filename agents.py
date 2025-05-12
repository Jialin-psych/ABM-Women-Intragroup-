from mesa import Agent

class Woman(Agent):
    """
    A Woman agent with gender role beliefs (GRB) and submission level (S).
    GRB influences submission behavior, and submission can vary around GRB.
    """

    def __init__(self, unique_id, model, grb):
        """
        Initializes a Woman agent with a unique ID, model, and gender role belief (GRB).
        
        Args:
            unique_id (int): Unique identifier for the agent.
            model (Model): The model the agent belongs to.
            grb (float): The agent's gender role belief (GRB), between 0 and 100.
        """
        super().__init__(unique_id, model)
        self.grb = grb  # Gender Role Beliefs (GRB) between 0 and 100
        self.submission_level = self.calculate_submission_level(grb)  # Initial submission level based on GRB

    def calculate_submission_level(self, grb):
        """
        Calculate the agent's submission level based on their GRB.
        The submission level can vary between GRB ± 10.

        Args:
            grb (float): The agent's gender role belief (GRB).
        
        Returns:
            float: The agent's initial submission level.
        """
        return random.uniform(grb - 10, grb + 10)  # Allowing submission to vary around GRB within ±10

    #def step(self):
      # Further code needded for the submission information spreading mechanisms

    def react_to_submission(self, other_agent):
        """
        Reacts to another agent's submission level by potentially expelling her.
        The agent expels another if their submission levels differ significantly.
        
        Args:
            other_agent (Woman): The other agent being interacted with.
        """
        if abs(self.submission_level - other_agent.submission_level) > 30:  # Threshold for perceiving threat
            self.expel(other_agent)

    def expel(self, other_agent):
        """
        Expels another agent from the group by removing the connection from the social network.
        If no connection exists, a negative signal is sent to the other agent.
        
        Args:
            other_agent (Woman): The agent being expelled.
        """
        # If the agents are connected in the social network, remove the edge
        if self.model.network.has_edge(self.unique_id, other_agent.unique_id):
            self.model.network.remove_edge(self.unique_id, other_agent.unique_id)  # Remove connection (edge)
        else:
            # If no connection exists, send a negative signal (for now, just updating a flag)
            self.send_negative_signal(other_agent)

    #def send_negative_signal(self, other_agent):
       # Still unsure about the mechanism of sending negative signals
