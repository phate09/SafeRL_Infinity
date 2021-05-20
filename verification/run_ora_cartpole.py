import os

import gurobi as grb
import ray
from pyomo.opt import SolverStatus, TerminationCondition

import utils
from verification.experiments_nn_analysis import Experiment
from verification.run_experiment_cartpole import CartpoleExperiment
import pyomo.environ as pyo

USE_GUROBI = True


class ORACartpoleExperiment(CartpoleExperiment):
    def __init__(self):
        super().__init__()
        self.post_fn_remote = self.post_milp
        self.before_start_fn = self.before_start
        self.time_horizon = 300
        self.use_rounding = False
        # self.nn_path = os.path.join(utils.get_save_dir(), "tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00001_1_cost_fn=1,tau=0.001_2021-01-16_20-25-43/checkpoint_3090/checkpoint-3090")
        self.nn_path = os.path.join(utils.get_save_dir(), "tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00000_0_cost_fn=0,tau=0.001_2021-01-16_20-25-43/checkpoint_193/checkpoint-193")

    def generate_nn_polyhedral_guard(self, nn, chosen_action, output_flag):
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', output_flag)
        gurobi_model.setParam('Threads', 2)
        observation = gurobi_model.addMVar(shape=(self.env_input_size,), lb=float("-inf"), ub=float("inf"), name="observation")
        Experiment.generate_nn_guard(gurobi_model, observation, nn, action_ego=chosen_action)
        # observable_template = self.get_template(1)
        observable_template = self.analysis_template
        # self.env_input_size = 2
        observable_result = self.optimise(observable_template, gurobi_model, observation)
        # self.env_input_size = 6
        return observable_template, observable_result

    def before_start(self, nn):
        observable_templates = []
        observable_results = []
        for chosen_action in range(2):
            observable_template, observable_result = self.generate_nn_polyhedral_guard(nn, chosen_action, False)
            observable_templates.append(observable_template)
            observable_results.append(observable_result)
        self.observable_templates = observable_templates
        self.observable_results = observable_results

    @ray.remote
    def post_milp(self, x, nn, output_flag, t, template):
        """milp method"""
        post = []
        for chosen_action in range(2):
            observable_template = self.observable_templates[chosen_action]
            observable_result = self.observable_results[chosen_action]
            if USE_GUROBI:
                gurobi_model = grb.Model()
                gurobi_model.setParam('OutputFlag', output_flag)
                gurobi_model.setParam('Threads', 2)
                input = Experiment.generate_input_region(gurobi_model, template, x, self.env_input_size)
                Experiment.generate_region_constraints(gurobi_model, observable_template, input, observable_result, env_input_size=self.env_input_size)
                gurobi_model.optimize()
                feasible_action = gurobi_model.status == 2
                if feasible_action:
                    max_theta, min_theta, max_theta_dot, min_theta_dot = self.get_theta_bounds(gurobi_model, input)
                    sin_cos_table = self.get_sin_cos_table(max_theta, min_theta, max_theta_dot, min_theta_dot, action=chosen_action, step_thetaacc=100)
                    x_prime_results = self.optimise(template, gurobi_model, input)  # h representation
                    x_prime = Experiment.generate_input_region(gurobi_model, template, x_prime_results, self.env_input_size)
                    thetaacc, xacc = CartpoleExperiment.generate_angle_milp(gurobi_model, x_prime, sin_cos_table)
                    # apply dynamic
                    x_second = self.apply_dynamic(x_prime, gurobi_model, thetaacc=thetaacc, xacc=xacc, env_input_size=self.env_input_size)
                    gurobi_model.update()
                    gurobi_model.optimize()
                    found_successor, x_second_results = self.h_repr_to_plot(gurobi_model, template, x_second)
                    if found_successor:
                        post.append(tuple(x_second_results))
            else:
                model = pyo.ConcreteModel()
                input = Experiment.generate_input_region_pyo(model, template, x, self.env_input_size)
                feasible_action = ORACartpoleExperiment.generate_nn_guard_pyo(model, input, nn, action_ego=chosen_action, M=1e04)
                if feasible_action:  # performs action 2 automatically when battery is dead
                    max_theta, min_theta, max_theta_dot, min_theta_dot = self.get_theta_bounds_pyo(model, input)
                    sin_cos_table = self.get_sin_cos_table(max_theta, min_theta, max_theta_dot, min_theta_dot, action=chosen_action, step_thetaacc=100)
                    x_prime_results = self.optimise_pyo(template, model, input)
                    x_prime = Experiment.generate_input_region_pyo(model, template, x_prime_results, self.env_input_size, name="x_prime_input")
                    thetaacc, xacc = ORACartpoleExperiment.generate_angle_milp_pyo(model, x_prime, sin_cos_table)

                    model.del_component(model.obj)
                    model.obj = pyo.Objective(expr=thetaacc, sense=pyo.maximize)
                    result = Experiment.solve(model, solver=Experiment.use_solver)
                    assert (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal), f"LP wasn't optimally solved {x}"
                    # apply dynamic
                    x_second = self.apply_dynamic_pyo(x_prime, model, thetaacc=thetaacc, xacc=xacc, env_input_size=self.env_input_size, action=chosen_action)
                    x_second_results = self.optimise_pyo(template, model, x_second)
                    found_successor = x_prime_results is not None
                    if found_successor:
                        post.append((tuple(x_second_results)))
        return post


if __name__ == '__main__':
    ray.init(log_to_driver=False, local_mode=True)
    experiment = ORACartpoleExperiment()
    # experiment.n_workers = 1
    experiment.run_experiment()
