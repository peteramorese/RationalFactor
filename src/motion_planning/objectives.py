from .controller import TrainableController
from .mdp import MarkovDecisionProcess
from rational_factor.models.density_model import DensityModel
from rational_factor.tools.set import Hyperrectangle


def probability_of_collision(init_belief: DensityModel, mdp: MarkovDecisionProcess, controller: TrainableController, obstacles: Hyperrectangle) -> float:
    belief_seq = [init_belief]
    for t in range(controller.n_timesteps()):
        belief = mdp.propagate_set_complement(belief_seq[t], controller(t), obstacles)
        belief_seq.append(belief)
    
    return belief_seq[-1].integrate(obstacles)

def probability_of_reach_avoid(init_belief: DensityModel, mdp: MarkovDecisionProcess, controller: TrainableController, obstacles: Hyperrectangle, goal: Hyperrectangle) -> float:
    goal_and_obstacles = obstacles.union(goal)

    belief_seq = [init_belief]
    goal_prob = init_belief.integrate(goal)
    for t in range(controller.n_timesteps()):
        belief = mdp.propagate_set_complement(belief_seq[t], controller(t), goal_and_obstacles)
        goal_prob += belief.integrate(goal)
        belief_seq.append(belief)
    
    return goal_prob