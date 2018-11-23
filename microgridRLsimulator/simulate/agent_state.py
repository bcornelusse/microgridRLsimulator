class AgentState:
    def __init__(self, gridstate):
        self.state = [gridstate.non_steerable_consumption, gridstate.state_of_charge, gridstate.non_steerable_production]