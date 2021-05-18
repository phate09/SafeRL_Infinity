import os
from typing import List, Tuple

from pyomo.opt import SolverStatus, TerminationCondition
from ray.rllib.agents.ppo import ppo

import utils
from training.tune_train_PPO_cartpole import get_PPO_config
from agents.ray_utils import convert_ray_policy_to_sequential
from verification.experiments_nn_analysis import Experiment
import ray
import gurobi as grb
import math
import numpy as np
import torch
from interval import interval, imath
from environments.cartpole_ray import CartPoleEnv
import pyomo.environ as pyo


class CartpoleExperiment(Experiment):
    def __init__(self):
        env_input_size: int = 4
        super().__init__(env_input_size)
        self.post_fn_remote = self.post_milp
        self.get_nn_fn = self.get_nn
        self.plot_fn = self.plot
        self.template_2d: np.ndarray = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        input_boundaries, input_template = self.get_template(0)
        self.input_boundaries: List = input_boundaries
        self.input_template: np.ndarray = input_template
        _, template = self.get_template(1)
        self.analysis_template: np.ndarray = template
        safe_angle = 12 * 2 * math.pi / 360
        theta = [self.e(4, 2)]
        neg_theta = [-self.e(4, 2)]
        self.unsafe_zone: List[Tuple] = [(theta, np.array([-safe_angle])), (neg_theta, np.array([-safe_angle]))]
        # self.use_rounding = False
        self.rounding_value = 1024
        self.time_horizon = 300
        self.nn_path = os.path.join(utils.get_save_dir(), "tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00001_1_cost_fn=1,tau=0.001_2021-01-16_20-25-43/checkpoint_3090/checkpoint-3090")
        # self.tau = 0.001
        self.tau = 0.02

    @ray.remote
    def post_milp(self, x, nn, output_flag, t, template):
        """milp method"""
        post = []
        for chosen_action in range(2):
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            gurobi_model.setParam('Threads', 2)
            input = Experiment.generate_input_region(gurobi_model, template, x, self.env_input_size)
            max_theta, min_theta, max_theta_dot, min_theta_dot = self.get_theta_bounds(gurobi_model, input)
            sin_cos_table = self.get_sin_cos_table(max_theta, min_theta, max_theta_dot, min_theta_dot, action=chosen_action)
            feasible_action = CartpoleExperiment.generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action)
            if feasible_action:
                thetaacc, xacc = CartpoleExperiment.generate_angle_milp(gurobi_model, input, sin_cos_table)
                # apply dynamic
                x_prime = self.apply_dynamic(input, gurobi_model, thetaacc=thetaacc, xacc=xacc, env_input_size=self.env_input_size)
                gurobi_model.update()
                gurobi_model.optimize()
                found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
                if found_successor:
                    post.append(tuple(x_prime_results))
        return post

    def apply_dynamic(self, input, gurobi_model: grb.Model, thetaacc, xacc, env_input_size):
        '''

        :param costheta: gurobi variable containing the range of costheta values
        :param sintheta: gurobi variable containin the range of sintheta values
        :param input:
        :param gurobi_model:
        :param t:
        :return:
        '''

        tau = self.tau  # 0.001  # seconds between state updates
        x = input[0]
        x_dot = input[1]
        theta = input[2]
        theta_dot = input[3]
        z = gurobi_model.addMVar(shape=(env_input_size,), lb=float("-inf"), name=f"x_prime")
        x_prime = x + tau * x_dot
        x_dot_prime = x_dot + tau * xacc
        theta_prime = theta + tau * theta_dot
        theta_dot_prime = theta_dot + tau * thetaacc
        gurobi_model.addConstr(z[0] == x_prime, name=f"dyna_constr_1")
        gurobi_model.addConstr(z[1] == x_dot_prime, name=f"dyna_constr_2")
        gurobi_model.addConstr(z[2] == theta_prime, name=f"dyna_constr_3")
        gurobi_model.addConstr(z[3] == theta_dot_prime, name=f"dyna_constr_4")
        return z

    def apply_dynamic_pyo(self, input, model: pyo.ConcreteModel, thetaacc, xacc, env_input_size, action):
        '''

        :param costheta: gurobi variable containing the range of costheta values
        :param sintheta: gurobi variable containin the range of sintheta values
        :param input:
        :param gurobi_model:
        :param t:
        :return:
        '''

        tau = self.tau  # 0.001  # seconds between state updates
        x = input[0]
        x_dot = input[1]
        theta = input[2]
        theta_dot = input[3]
        z = pyo.Var(range(env_input_size), name=f"x_prime", within=pyo.Reals)
        model.add_component("x_prime", z)
        x_prime = x + tau * x_dot
        x_dot_prime = x_dot + tau * xacc
        theta_prime = theta + tau * theta_dot
        theta_dot_prime = theta_dot + tau * thetaacc
        model.dynamic_constraints = pyo.ConstraintList()
        model.dynamic_constraints.add(expr=z[0] == x_prime)
        model.dynamic_constraints.add(expr=z[1] == x_dot_prime)
        model.dynamic_constraints.add(expr=z[2] == theta_prime)
        model.dynamic_constraints.add(expr=z[3] == theta_dot_prime)
        return z

    @staticmethod
    def get_sin_cos_table(max_theta, min_theta, max_theta_dot, min_theta_dot, action, step_thetaacc=0.3):
        assert min_theta <= max_theta, f"min_theta = {min_theta},max_theta={max_theta}"
        assert min_theta_dot <= max_theta_dot, f"min_theta_dot = {min_theta_dot},max_theta_dot={max_theta_dot}"
        step_theta = 0.1
        step_theta_dot = 0.1
        min_theta = max(min_theta, -math.pi / 2)
        max_theta = min(max_theta, math.pi / 2)
        split_theta1 = np.arange(min(min_theta, 0), min(max_theta, 0), step_theta)
        split_theta2 = np.arange(max(min_theta, 0), max(max_theta, 0), step_theta)
        split_theta = np.concatenate([split_theta1, split_theta2])
        split_theta_dot1 = np.arange(min(min_theta_dot, 0), min(max_theta_dot, 0), step_theta)
        split_theta_dot2 = np.arange(max(min_theta_dot, 0), max(max_theta_dot, 0), step_theta)
        split_theta_dot = np.concatenate([split_theta_dot1, split_theta_dot2])
        env = CartPoleEnv(None)
        force = env.force_mag if action == 1 else -env.force_mag

        split = []
        for t_dot in split_theta_dot:
            for theta in split_theta:
                lb_theta_dot = t_dot
                ub_theta_dot = min(t_dot + step_theta_dot, max_theta_dot)
                lb = theta
                ub = min(theta + step_theta, max_theta)
                split.append(((min(lb_theta_dot, ub_theta_dot), max(lb_theta_dot, ub_theta_dot)), (min(lb, ub), max(lb, ub))))
        sin_cos_table = []
        while (len(split)):
            theta_dot, theta = split.pop()
            theta_interval = interval([theta[0], theta[1]])
            theta_dot_interval = interval([theta_dot[0], theta_dot[1]])
            sintheta = imath.sin(theta_interval)
            costheta = imath.cos(theta_interval)
            temp = (force + env.polemass_length * theta_dot_interval ** 2 * sintheta) / env.total_mass
            thetaacc: interval = (env.gravity * sintheta - costheta * temp) / (env.length * (4.0 / 3.0 - env.masspole * costheta ** 2 / env.total_mass))
            xacc = temp - env.polemass_length * thetaacc * costheta / env.total_mass
            if thetaacc[0].sup - thetaacc[0].inf > step_thetaacc:
                # split theta theta_dot
                if (theta[1] - theta[0]) > (theta_dot[1] - theta_dot[0]):  # split theta
                    mid_theta = (theta[0] + theta[1]) / 2
                    theta_1 = (theta[0], mid_theta)
                    theta_2 = (mid_theta, theta[1])
                    split.append((theta_1, theta_dot))
                    split.append((theta_2, theta_dot))
                else:  # split theta_dot
                    mid_theta_dot = (theta_dot[1] + theta_dot[0]) / 2
                    theta_dot_1 = (theta_dot[0], mid_theta_dot)
                    theta_dot_2 = (mid_theta_dot, theta_dot[1])
                    split.append((theta, theta_dot_1))
                    split.append((theta, theta_dot_2))
            else:
                sin_cos_table.append((theta, theta_dot,  (thetaacc[0].inf,thetaacc[0].sup), (xacc[0].inf,xacc[0].sup)))
        return sin_cos_table

    @staticmethod
    def get_theta_bounds(gurobi_model, input):
        gurobi_model.setObjective(input[2].sum(), grb.GRB.MAXIMIZE)
        gurobi_model.optimize()
        max_theta = gurobi_model.getVars()[2].X

        gurobi_model.setObjective(input[2].sum(), grb.GRB.MINIMIZE)
        gurobi_model.optimize()
        min_theta = gurobi_model.getVars()[2].X

        gurobi_model.setObjective(input[3].sum(), grb.GRB.MAXIMIZE)
        gurobi_model.optimize()
        max_theta_dot = gurobi_model.getVars()[3].X

        gurobi_model.setObjective(input[3].sum(), grb.GRB.MINIMIZE)
        gurobi_model.optimize()
        min_theta_dot = gurobi_model.getVars()[3].X
        return max_theta, min_theta, max_theta_dot, min_theta_dot

    @staticmethod
    def get_theta_bounds_pyo(model: pyo.ConcreteModel, input):
        if model.component("obj"):
            model.del_component(model.obj)
        model.obj = pyo.Objective(expr=input[2], sense=pyo.maximize)
        result = Experiment.solve(model, solver=Experiment.use_solver)
        assert (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal)
        max_theta = pyo.value(model.obj)

        model.del_component(model.obj)
        model.obj = pyo.Objective(expr=input[2], sense=pyo.minimize)
        result = Experiment.solve(model, solver=Experiment.use_solver)
        assert (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal)
        min_theta = pyo.value(model.obj)

        model.del_component(model.obj)
        model.obj = pyo.Objective(expr=input[3], sense=pyo.maximize)
        result = Experiment.solve(model, solver=Experiment.use_solver)
        assert (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal)
        max_theta_dot = pyo.value(model.obj)

        model.del_component(model.obj)
        model.obj = pyo.Objective(expr=input[3], sense=pyo.minimize)
        result = Experiment.solve(model, solver=Experiment.use_solver)
        assert (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal)
        min_theta_dot = pyo.value(model.obj)
        return max_theta, min_theta, max_theta_dot, min_theta_dot

    @staticmethod
    def generate_angle_milp(gurobi_model: grb.Model, input, sin_cos_table: List[Tuple]):
        """MILP method
        input: theta, thetadot
        output: thetadotdot, xdotdot (edited)
        l_{theta, i}, l_{thatdot,i}, l_{thetadotdot, i}, l_{xdotdot, i}, u_....
        sum_{i=1}^k l_{x,i} - l_{x,i}*z_i <= x <= sum_{i=1}^k u_{x,i} - u_{x,i}*z_i, for each variable x
        sum_{i=1}^k l_{theta,i} - l_{theta,i}*z_i <= theta <= sum_{i=1}^k u_{theta,i} - u_{theta,i}*z_i
        """
        theta = input[2]
        theta_dot = input[3]
        k = len(sin_cos_table)
        zs = []
        thetaacc = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name="thetaacc")
        xacc = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name="xacc")
        for i in range(k):
            z = gurobi_model.addMVar(lb=0, ub=1, shape=(1,), vtype=grb.GRB.INTEGER, name=f"part_{i}")
            zs.append(z)
        gurobi_model.addConstr(k - 1 == sum(zs), name=f"const_milp1")
        theta_lb = 0
        theta_ub = 0
        theta_dot_lb = 0
        theta_dot_ub = 0
        thetaacc_lb = 0
        thetaacc_ub = 0
        xacc_lb = 0
        xacc_ub = 0
        for i in range(k):
            theta_interval, theta_dot_interval, theta_acc_interval, xacc_interval = sin_cos_table[i]
            theta_lb += theta_interval[0] - theta_interval[0] * zs[i]
            theta_ub += theta_interval[1] - theta_interval[1] * zs[i]
            theta_dot_lb += theta_dot_interval[0] - theta_dot_interval[0] * zs[i]
            theta_dot_ub += theta_dot_interval[1] - theta_dot_interval[1] * zs[i]

            thetaacc_lb += theta_acc_interval[0] - theta_acc_interval[0] * zs[i]
            thetaacc_ub += theta_acc_interval[1] - theta_acc_interval[1] * zs[i]

            xacc_lb += xacc_interval[0] - xacc_interval[0] * zs[i]
            xacc_ub += xacc_interval[1] - xacc_interval[1] * zs[i]
        # eps = 1e-9
        gurobi_model.addConstr(theta >= theta_lb, name=f"theta_guard1")
        gurobi_model.addConstr(theta <= theta_ub, name=f"theta_guard2")
        gurobi_model.addConstr(theta_dot >= theta_dot_lb, name=f"theta_dot_guard1")
        gurobi_model.addConstr(theta_dot <= theta_dot_ub, name=f"theta_dot_guard2")

        gurobi_model.addConstr(thetaacc >= thetaacc_lb, name=f"thetaacc_guard1")
        gurobi_model.addConstr(thetaacc <= thetaacc_ub, name=f"thetaacc_guard2")
        gurobi_model.addConstr(xacc >= xacc_lb, name=f"xacc_guard1")
        gurobi_model.addConstr(xacc <= xacc_ub, name=f"xacc_guard2")

        gurobi_model.update()
        gurobi_model.optimize()
        if gurobi_model.status == 4:
            gurobi_model.setParam("DualReductions", 0)
            gurobi_model.update()
            gurobi_model.optimize()
        assert gurobi_model.status == 2, f"LP wasn't optimally solved. gurobi status {gurobi_model.status}"
        return thetaacc, xacc

    @staticmethod
    def generate_angle_milp_pyo(model: pyo.ConcreteModel, input, sin_cos_table: List[Tuple]):
        """MILP method
        input: theta, thetadot
        output: thetadotdot, xdotdot (edited)
        l_{theta, i}, l_{thatdot,i}, l_{thetadotdot, i}, l_{xdotdot, i}, u_....
        sum_{i=1}^k l_{x,i} - l_{x,i}*z_i <= x <= sum_{i=1}^k u_{x,i} - u_{x,i}*z_i, for each variable x
        sum_{i=1}^k l_{theta,i} - l_{theta,i}*z_i <= theta <= sum_{i=1}^k u_{theta,i} - u_{theta,i}*z_i
        """
        # gurobi_model.setParam('OptimalityTol', 1e-6)
        # gurobi_model.setParam('FeasibilityTol', 1e-6)
        # gurobi_model.setParam('IntFeasTol', 1e-9)
        theta = input[2]
        theta_dot = input[3]
        k = len(sin_cos_table)
        pyo.Var()
        thetaacc = pyo.Var(name=f"thetaacc", within=pyo.Reals)
        model.add_component("thetaacc", thetaacc)
        xacc = pyo.Var(name=f"xacc", within=pyo.Reals)
        model.add_component("xacc", xacc)
        zs = pyo.Var(range(k), name=f"z", within=pyo.Integers, bounds=(0, 1))
        model.add_component("angle_section", zs)
        model.angle_constraints = pyo.ConstraintList()
        model.angle_constraints.add(expr=k - 1 == sum([zs[i] for i in range(k)]))
        theta_lb = 0
        theta_ub = 0
        theta_dot_lb = 0
        theta_dot_ub = 0
        thetaacc_lb = 0
        thetaacc_ub = 0
        xacc_lb = 0
        xacc_ub = 0
        round_value = 2 ** 20
        for i in range(k):
            theta_interval, theta_dot_interval, theta_acc_interval, xacc_interval = sin_cos_table[i]
            theta_inf = theta_interval[0].inf
            theta_sup = theta_interval[0].sup
            theta_dot_inf = theta_dot_interval[0].inf
            theta_dot_sup = theta_dot_interval[0].sup
            theta_lb += theta_inf - theta_inf * zs[i]
            theta_ub += theta_sup - theta_sup * zs[i]
            theta_dot_lb += theta_dot_inf - theta_dot_inf * zs[i]
            theta_dot_ub += theta_dot_sup - theta_dot_sup * zs[i]

            theta_acc_interval__inf = theta_acc_interval[0].inf
            theta_acc_interval__sup = theta_acc_interval[0].sup
            xacc_interval__inf = xacc_interval[0].inf
            xacc_interval__sup = xacc_interval[0].sup

            thetaacc_lb += theta_acc_interval__inf - theta_acc_interval__inf * zs[i]
            thetaacc_ub += theta_acc_interval__sup - theta_acc_interval__sup * zs[i]
            xacc_lb += xacc_interval__inf - xacc_interval__inf * zs[i]
            xacc_ub += xacc_interval__sup - xacc_interval__sup * zs[i]

        model.angle_constraints.add(expr=theta >= theta_lb)
        model.angle_constraints.add(expr=theta <= theta_ub)
        model.angle_constraints.add(expr=theta_dot >= theta_dot_lb)
        model.angle_constraints.add(expr=theta_dot <= theta_dot_ub)

        model.angle_constraints.add(expr=thetaacc >= thetaacc_lb)
        model.angle_constraints.add(expr=thetaacc <= thetaacc_ub)
        model.angle_constraints.add(expr=xacc >= xacc_lb)
        model.angle_constraints.add(expr=xacc <= xacc_ub)

        return thetaacc, xacc

    def plot(self, vertices_list, template, template_2d):
        self.generic_plot("theta", "theta_dot", vertices_list, template, template_2d)

    def get_template(self, mode=0):
        x = Experiment.e(self.env_input_size, 0)
        x_dot = Experiment.e(self.env_input_size, 1)
        theta = Experiment.e(self.env_input_size, 2)
        theta_dot = Experiment.e(self.env_input_size, 3)
        if mode == 0:  # box directions with intervals
            # input_boundaries = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
            input_boundaries = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
            # input_boundaries = [0.04373426, -0.04373426, -0.04980056, 0.04980056, 0.045, -0.045, -0.51, 0.51]
            # optimise in a direction
            template = []
            for dimension in range(self.env_input_size):
                template.append(Experiment.e(self.env_input_size, dimension))
                template.append(-Experiment.e(self.env_input_size, dimension))
            template = np.array(template)  # the 6 dimensions in 2 variables
            return input_boundaries, template
        if mode == 1:  # directions to easily find fixed point
            input_boundaries = None
            template = np.array([theta, -theta, theta_dot, -theta_dot, theta + theta_dot, -(theta + theta_dot), (theta - theta_dot), -(theta - theta_dot)])  # x_dot, -x_dot,theta_dot - theta
            return input_boundaries, template
        if mode == 2:
            input_boundaries = None
            template = np.array([theta, -theta, theta_dot, -theta_dot])
            return input_boundaries, template
        if mode == 3:
            input_boundaries = None
            template = np.array([theta, theta_dot, -theta_dot])
            return input_boundaries, template
        if mode == 4:
            input_boundaries = [0.09375, 0.625, 0.625, 0.0625, 0.1875]
            # input_boundaries = [0.09375, 0.5, 0.5, 0.0625, 0.09375]
            template = np.array([theta, theta_dot, -theta_dot, theta + theta_dot, (theta - theta_dot)])
            return input_boundaries, template
        if mode == 5:
            input_boundaries = [0.125, 0.0625, 0.1875]
            template = np.array([theta, theta + theta_dot, (theta - theta_dot)])
            return input_boundaries, template

    def get_nn(self):
        config = get_PPO_config(1234)
        trainer = ppo.PPOTrainer(config=config)
        trainer.restore(self.nn_path)

        policy = trainer.get_policy()
        # sequential_nn = convert_ray_simple_policy_to_sequential(policy).cpu()
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
        l0 = torch.nn.Linear(4, 2, bias=False)
        l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32))
        layers = [l0]
        for l in sequential_nn:
            layers.append(l)
        nn = torch.nn.Sequential(*layers)
        nn.double()
        # ray.shutdown()
        return nn


if __name__ == '__main__':
    ray.init()
    experiment = CartpoleExperiment()
    experiment.run_experiment()
