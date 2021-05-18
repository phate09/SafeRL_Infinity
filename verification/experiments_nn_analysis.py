import csv
import datetime
import math
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Tuple, List

import gurobi as grb
import numpy as np
import progressbar
import ray
import torch
from pyomo.core import TransformationFactory
from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.environ as pyo
import pyomo.gdp as gdp

from verification.plot_utils import show_polygon_list3


class Experiment():
    def __init__(self, env_input_size: int):
        self.before_start_fn = None
        self.update_progress_fn = None
        self.rounding_value = 1024
        self.get_nn_fn = None
        self.plot_fn = None
        self.post_fn_remote = None
        self.template_2d: np.ndarray = None
        self.input_template: np.ndarray = None
        self.input_boundaries: List = None
        self.analysis_template: np.ndarray = None
        self.unsafe_zone: List[Tuple] = None
        self.output_flag = False
        self.env_input_size: int = env_input_size
        self.n_workers = 8
        self.plotting_time_interval = 60 * 2
        self.time_horizon = 100
        self.use_bfs = True  # use Breadth-first-search or Depth-first-search
        self.local_mode = False  # disable multi processing
        self.use_rounding = True
        self.show_progressbar = True
        self.show_progress_plot = True
        self.save_dir = None

    use_solver = "glpk"

    @staticmethod
    def solve(model, solver="glpk"):
        if solver == "glpk":
            return pyo.SolverFactory('glpk', algorithm="exact").solve(model)
        elif solver == "gurobi":
            return pyo.SolverFactory('gurobi', solver_io="python").solve(model)
        elif solver == "gurobi2":
            return pyo.SolverFactory('gurobi_direct').solve(model)
        elif solver == "cplex":
            return pyo.SolverFactory('cplex_direct').solve(model)
        elif solver == "cbc":
            return pyo.SolverFactory('cbc').solve(model)
        else:
            raise NotImplementedError()

    def run_experiment(self):
        assert self.get_nn_fn is not None
        assert self.plot_fn is not None
        assert self.post_fn_remote is not None
        assert self.template_2d is not None
        assert self.input_template is not None
        assert self.input_boundaries is not None
        assert self.analysis_template is not None
        assert self.unsafe_zone is not None
        experiment_start_time = time.time()
        nn: torch.nn.Sequential = self.get_nn_fn()
        root = self.generate_root_polytope()
        max_t, num_already_visited, vertices_list, unsafe = self.main_loop(nn, self.analysis_template, [root], self.template_2d)
        print(f"T={max_t}")

        print(f"The algorithm skipped {num_already_visited} already visited states")
        safe = None
        if unsafe:
            print("The agent is unsafe")
            safe = False
        elif max_t < self.time_horizon:
            print("The agent is safe")
            safe = True
        else:
            print(f"It could not be determined if the agent is safe or not within {self.time_horizon} steps. Increase 'time_horizon' to increase the number of steps to analyse")
            safe = None
        experiment_end_time = time.time()
        elapsed_seconds = round((experiment_end_time - experiment_start_time))
        print(f"Total verification time {str(datetime.timedelta(seconds=elapsed_seconds))}")
        return elapsed_seconds, safe, max_t

    def main_loop(self, nn, template, root_list: List[Tuple], template_2d):
        vertices_list = defaultdict(list)
        seen = []
        frontier = [(0, x) for x in root_list]
        max_t = 0
        num_already_visited = 0
        widgets = [progressbar.Variable('n_workers'), ', ', progressbar.Variable('frontier'), ', ', progressbar.Variable('seen'), ', ', progressbar.Variable('num_already_visited'), ", ",
                   progressbar.Variable('max_t'), ", ", progressbar.Variable('last_visited_state')]
        proc_ids = []
        last_time_plot = None
        if self.before_start_fn is not None:
            self.before_start_fn(nn)
        with progressbar.ProgressBar(widgets=widgets) if self.show_progressbar else nullcontext() as bar:
            while len(frontier) != 0 or len(proc_ids) != 0:
                while len(proc_ids) < self.n_workers and len(frontier) != 0:
                    t, x = frontier.pop(0) if self.use_bfs else frontier.pop()
                    if max_t > self.time_horizon:
                        print(f"Reached horizon t={t}")
                        self.plot_fn(vertices_list, template, template_2d)
                        return max_t, num_already_visited, vertices_list, False
                    contained_flag = False
                    to_remove = []
                    for s in seen:
                        if contained(x, s):
                            contained_flag = True
                            break
                        if contained(s, x):
                            to_remove.append(s)
                    for rem in to_remove:
                        num_already_visited += 1
                        seen.remove(rem)
                    if contained_flag:
                        num_already_visited += 1
                        continue
                    max_t = max(max_t, t)
                    vertices_list[t].append(np.array(x))
                    if self.check_unsafe(template, x):
                        print(f"Unsafe state found at timestep t={t}")
                        print(x)
                        self.plot_fn(vertices_list, template, template_2d)
                        return max_t, num_already_visited, vertices_list, True
                    seen.append(x)
                    proc_ids.append(self.post_fn_remote.remote(self, x, nn, self.output_flag, t, template))
                if last_time_plot is None or time.time() - last_time_plot >= self.plotting_time_interval:
                    if last_time_plot is not None:
                        self.plot_fn(vertices_list, template, template_2d)
                    last_time_plot = time.time()
                if self.update_progress_fn is not None:
                    self.update_progress_fn(n_workers=len(proc_ids), seen=len(seen), frontier=len(frontier), num_already_visited=num_already_visited, max_t=max_t)
                if self.show_progressbar:
                    bar.update(value=bar.value + 1, n_workers=len(proc_ids), seen=len(seen), frontier=len(frontier), num_already_visited=num_already_visited, last_visited_state=str(x), max_t=max_t)
                ready_ids, proc_ids = ray.wait(proc_ids, num_returns=len(proc_ids), timeout=0.5)
                x_primes_list = ray.get(ready_ids)
                for x_primes in x_primes_list:
                    for x_prime in x_primes:
                        if self.use_rounding:
                            x_prime_rounded = self.round_tuple(x_prime, self.rounding_value)
                            # x_prime_rounded should always be bigger than x_prime
                            assert contained(x_prime, x_prime_rounded)
                            x_prime = x_prime_rounded
                        frontier = [(u, y) for u, y in frontier if not contained(y, x_prime)]
                        if not any([contained(x_prime, y) for u, y in frontier]):
                            frontier.append(((t + 1), x_prime))
                            # print(x_prime)
                        else:
                            num_already_visited += 1
        self.plot_fn(vertices_list, template, template_2d)
        return max_t, num_already_visited, vertices_list, False

    def round_tuple(self, x, rounding_value):
        rounded_x = []
        for val in x:
            if val < 0:
                rounded_x.append(-1 * math.floor(abs(val) * rounding_value) / rounding_value)
            else:
                rounded_x.append(math.ceil(abs(val) * rounding_value) / rounding_value)
        return tuple(rounded_x)

    def generate_root_polytope(self):
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', self.output_flag)
        input = Experiment.generate_input_region(gurobi_model, self.input_template, self.input_boundaries, self.env_input_size)
        x_results = self.optimise(self.analysis_template, gurobi_model, input)
        if x_results is None:
            print("Model unsatisfiable")
            return None
        root = tuple(x_results)
        return root

    @staticmethod
    def generate_input_region(gurobi_model, templates, boundaries, env_input_size):
        input = gurobi_model.addMVar(shape=env_input_size, lb=float("-inf"), ub=float("inf"), name="input")
        Experiment.generate_region_constraints(gurobi_model, templates, input, boundaries, env_input_size)
        return input

    @staticmethod
    def generate_input_region_pyo(model: pyo.ConcreteModel, templates, boundaries, env_input_size, name="input"):
        input = pyo.Var(range(env_input_size), domain=pyo.Reals, name=name)
        model.add_component(name=name, val=input)
        Experiment.generate_region_constraints_pyo(model, templates, input, boundaries, env_input_size)
        return model.input

    @staticmethod
    def generate_region_constraints(gurobi_model, templates, input, boundaries, env_input_size):
        for j, template in enumerate(templates):
            gurobi_model.update()
            multiplication = 0
            for i in range(env_input_size):
                multiplication += template[i] * input[i]
            gurobi_model.addConstr(multiplication <= boundaries[j], name=f"input_constr_{j}")

    @staticmethod
    def generate_region_constraints_pyo(model: pyo.ConcreteModel, templates, input, boundaries, env_input_size, invert=False):

        if model.component("region_constraints") is None:
            model.region_constraints = pyo.ConstraintList()
        for j, template in enumerate(templates):
            multiplication = 0
            for i in range(env_input_size):
                multiplication += template[i] * input[i]
            if not invert:
                # gurobi_model.addConstr(multiplication <= boundaries[j], name=f"input_constr_{j}")
                model.region_constraints.add(multiplication <= boundaries[j])
            else:
                # gurobi_model.addConstr(multiplication >= boundaries[j], name=f"input_constr_{j}")
                model.region_constraints.add(multiplication >= boundaries[j])

    def optimise(self, templates: np.ndarray, gurobi_model: grb.Model, x_prime: tuple):
        results = []
        for template in templates:
            gurobi_model.update()
            gurobi_model.setObjective(sum((template[i] * x_prime[i]) for i in range(self.env_input_size)), grb.GRB.MAXIMIZE)
            gurobi_model.optimize()
            # print_model(gurobi_model)
            if gurobi_model.status != 2:
                return None
            result = gurobi_model.ObjVal
            results.append(result)
        return np.array(results)

    def optimise_pyo(self, templates: np.ndarray, model: pyo.ConcreteModel, x_prime):
        results = []
        for template in templates:
            model.del_component(model.obj)
            model.obj = pyo.Objective(expr=sum((template[i] * x_prime[i]) for i in range(self.env_input_size)), sense=pyo.maximize)
            result = Experiment.solve(model, solver=Experiment.use_solver)
            # assert (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal), f"LP wasn't optimally solved {x}"
            # print_model(gurobi_model)
            if (result.solver.status == SolverStatus.ok):
                if (result.solver.termination_condition == TerminationCondition.optimal):
                    result = pyo.value(model.obj)
                    results.append(result)
                elif (result.solver.termination_condition == TerminationCondition.unbounded):
                    result = float("inf")
                    results.append(result)
                    continue
                else:
                    return None
            else:
                return None
        return np.array(results)

    def check_unsafe(self, template, bnds):
        for A, b in self.unsafe_zone:
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', False)
            input = gurobi_model.addMVar(shape=(self.env_input_size,), lb=float("-inf"), name="input")
            Experiment.generate_region_constraints(gurobi_model, template, input, bnds, self.env_input_size)
            Experiment.generate_region_constraints(gurobi_model, A, input, b, self.env_input_size)
            gurobi_model.update()
            gurobi_model.optimize()
            if gurobi_model.status == 2:
                return True
        return False

    @staticmethod
    def e(n, i):
        result = [0] * n
        result[i] = 1
        return np.array(result)

    @staticmethod
    def octagon(n):
        template = []
        for i in range(n):
            x = Experiment.e(n, i)
            template.append(x)
            template.append(-x)
            for j in range(0, i):
                y = Experiment.e(n, j)
                template.append(x + y)
                template.append(x - y)
                template.append(y - x)
                template.append(-y - x)
        return np.stack(template)

    @staticmethod
    def box(n):
        template = []
        for i in range(n):
            x = Experiment.e(n, i)
            template.append(x)
            template.append(-x)
        return np.stack(template)

    def h_repr_to_plot(self, gurobi_model, template, x_prime):
        x_prime_results = self.optimise(template, gurobi_model, x_prime)  # h representation
        return x_prime_results is not None, x_prime_results

    @staticmethod
    def generate_nn_guard(gurobi_model: grb.Model, input, nn: torch.nn.Sequential, action_ego=0, M=1e6):
        gurobi_vars = []
        gurobi_vars.append(input)
        for i, layer in enumerate(nn):

            # print(layer)
            if type(layer) is torch.nn.Linear:
                v = gurobi_model.addMVar(lb=float("-inf"), shape=(layer.out_features), name=f"layer_{i}")
                weights: np.ndarray = layer.weight.data.numpy()
                weights.round(6)
                lin_expr = weights @ gurobi_vars[-1]
                if layer.bias is not None:
                    lin_expr = lin_expr + layer.bias.data.numpy()
                gurobi_model.addConstr(v == lin_expr, name=f"linear_constr_{i}")
                gurobi_vars.append(v)
            elif type(layer) is torch.nn.ReLU:
                v = gurobi_model.addMVar(lb=float("-inf"), shape=gurobi_vars[-1].shape, name=f"layer_{i}")  # same shape as previous
                z = gurobi_model.addMVar(lb=0, ub=1, shape=gurobi_vars[-1].shape, vtype=grb.GRB.INTEGER, name=f"relu_{i}")
                eps = 0  # 1e-9
                # gurobi_model.addConstr(v == grb.max_(0, gurobi_vars[-1]))
                gurobi_model.addConstr(v >= gurobi_vars[-1], name=f"relu_constr_1_{i}")
                gurobi_model.addConstr(v <= eps + gurobi_vars[-1] + M * z, name=f"relu_constr_2_{i}")
                gurobi_model.addConstr(v >= 0, name=f"relu_constr_3_{i}")
                gurobi_model.addConstr(v <= eps + M - M * z, name=f"relu_constr_4_{i}")
                gurobi_vars.append(v)
                # gurobi_model.update()
                # gurobi_model.optimize()
                # assert gurobi_model.status == 2, "LP wasn't optimally solved"
                """
                y = Relu(x)
                0 <= z <= 1, z is integer
                y >= x
                y <= x + Mz
                y >= 0
                y <= M - Mz"""
        # gurobi_model.update()
        # gurobi_model.optimize()
        # assert gurobi_model.status == 2, "LP wasn't optimally solved"
        # gurobi_model.setObjective(v[action_ego].sum(), grb.GRB.MAXIMIZE)  # maximise the output
        last_layer = gurobi_vars[-1]
        if action_ego == 0:
            gurobi_model.addConstr(last_layer[0] >= last_layer[1], name="last_layer")
        else:
            gurobi_model.addConstr(last_layer[1] >= last_layer[0], name="last_layer")
        gurobi_model.update()
        gurobi_model.optimize()
        # assert gurobi_model.status == 2, f"LP wasn't optimally solved, gurobi status {gurobi_model.status}"
        return gurobi_model.status == 2

    @staticmethod
    def generate_nn_guard_pyo(model: pyo.ConcreteModel, input, nn: torch.nn.Sequential, action_ego=0, M=1e2):
        model.nn_contraints = pyo.ConstraintList()
        gurobi_vars = []
        gurobi_vars.append(input)
        for i, layer in enumerate(nn):
            if type(layer) is torch.nn.Linear:
                layer_size = int(layer.out_features)
                v = pyo.Var(range(layer_size), name=f"layer_{i}", within=pyo.Reals)
                model.add_component(name=f"layer_{i}", val=v)
                lin_expr = np.zeros(layer_size)
                weights = layer.weight.data.numpy()
                bias = 0
                if layer.bias is not None:
                    bias = layer.bias.data.numpy()
                else:
                    bias = np.zeros(layer_size)
                for j in range(layer_size):
                    res = sum(gurobi_vars[-1][k] * weights[j, k] for k in range(weights.shape[1])) + bias[j]

                for j in range(layer_size):
                    model.nn_contraints.add(v[j] == sum(gurobi_vars[-1][k] * weights[j, k] for k in range(weights.shape[1])) + bias[j])
                gurobi_vars.append(v)
            elif type(layer) is torch.nn.ReLU:
                layer_size = int(nn[i - 1].out_features)
                v = pyo.Var(range(layer_size), name=f"layer_{i}", within=pyo.PositiveReals)
                model.add_component(name=f"layer_{i}", val=v)

                z = pyo.Var(range(layer_size), name=f"relu_{i}", within=pyo.Binary)
                model.add_component(name=f"relu_{i}", val=z)
                # for j in range(layer_size):
                #     model.nn_contraints.add(expr=v[j] >= gurobi_vars[-1][j])
                #     model.nn_contraints.add(expr=v[j] <= gurobi_vars[-1][j] + M * z[j])
                #     model.nn_contraints.add(expr=v[j] >= 0)
                #     model.nn_contraints.add(expr=v[j] <= M - M * z[j])

                for j in range(layer_size):
                    # model.nn_contraints.add(expr=v[j] <= gurobi_vars[-1][j])
                    dis = gdp.Disjunction(expr=[[v[j] >= gurobi_vars[-1][j], v[j] <= gurobi_vars[-1][j], gurobi_vars[-1][j] >= 0], [v[j] == 0, gurobi_vars[-1][j] <= 0]])
                    model.add_component(f"relu_{i}_{j}", dis)
                gurobi_vars.append(v)
                """
                y = Relu(x)
                0 <= z <= 1, z is integer
                y >= x
                y <= x + Mz
                y >= 0
                y <= M - Mz"""
        for i in range(len(gurobi_vars[-1])):
            if i == action_ego:
                continue
            model.nn_contraints.add(gurobi_vars[-1][action_ego] >= gurobi_vars[-1][i])
        if model.component("obj"):
            model.del_component("obj")
        model.obj = pyo.Objective(expr=gurobi_vars[-1][action_ego], sense=pyo.minimize)
        TransformationFactory('gdp.bigm').apply_to(model, bigM=M)
        result = Experiment.solve(model, solver=Experiment.use_solver)
        if (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal):
            return True
        elif (result.solver.termination_condition == TerminationCondition.infeasible or result.solver.termination_condition == TerminationCondition.infeasibleOrUnbounded):
            # log_infeasible_constraints(model)
            return False
        else:
            print(f"Solver status: {result.solver.status}")
            return False

    def generic_plot(self, title_x, title_y, vertices_list, template, template_2d):
        fig, simple_vertices = show_polygon_list3(vertices_list, title_x, title_y, template, template_2d)
        if self.show_progress_plot:
            fig.show()
        if self.save_dir is not None:
            width = 2560
            height = 1440
            scale = 1
            fig.write_image(os.path.join(self.save_dir, "plot.svg"), width=width, height=height, scale=scale)
            fig.write_image(os.path.join(self.save_dir, "plot.png"), width=width, height=height, scale=scale)
            fig.write_image(os.path.join(self.save_dir, "plot.jpeg"), width=width, height=height, scale=scale)
            fig.write_image(os.path.join(self.save_dir, "plot.pdf"), width=width, height=height, scale=scale)
            fig.write_html(os.path.join(self.save_dir, "plot.html"), include_plotlyjs="cdn")
            with open(os.path.join(self.save_dir, "plot.csv"), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
                for timestep in simple_vertices:
                    for item in timestep:
                        # assert len(item) == 4
                        for vertex in item:
                            wr.writerow(vertex)
                        wr.writerow(item[0])  # write back the first item
                    wr.writerow("")


def contained(x: tuple, y: tuple):
    # y contains x
    assert len(x) == len(y)
    for i in range(len(x)):
        if x[i] > y[i]:
            return False
    return True
