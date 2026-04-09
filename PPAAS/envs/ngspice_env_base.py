import gymnasium
from gymnasium import spaces
import numpy as np
import random
from collections import OrderedDict
import yaml
import yaml.constructor
import os
from eval_engines.util.core import *
import pickle
import os
import pdb
import copy
from abc import ABC, abstractmethod
from eval_engines.ngspice.CircuitClass import *
from typing import Union, List, Dict, Tuple


# way of ordering the way a yaml file is read
class OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.
    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        self.add_constructor("tag:yaml.org,2002:map", type(self).construct_yaml_map)
        self.add_constructor("tag:yaml.org,2002:omap", type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "expected a mapping node, but found %s" % node.id,
                node.start_mark,
            )

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping


class ngspice_env(gymnasium.Env, ABC):
    metadata = {"render.modes": ["human"]}

    ACT_LOW = -1
    ACT_HIGH = 1

    # obtains yaml file
    path = os.getcwd()
    CIR_YAML = (
        path + "/eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp.yaml"
    )

    def __init__(self, env_config):
        self.generalize = env_config.get("generalize", True)
        self.episode_len = env_config.get("episode_len", 30)
        self.valid = env_config.get("run_valid", False)
        self.CIR_YAML = env_config.get(
            "CIR_YAML", ngspice_env.CIR_YAML
        )  # path to the yaml file for evaluation

        # # Path to the pre-fixed target specs file for training
        self.spec_path = env_config.get(
            "spec_path", "ngspice_specs_gen_two_stage_opamp"
        )

        # Skip on Fail
        self.SoF = env_config.get("SoF", True)

        # lookup style for normalizing specs, which can be "normd" or "tanh"
        # normd is a normalization method we designed that normalizes the spec based on the desired spec (goal) and the current spec value,
        # and it has a nice property that when the current spec is close to the desired spec, the normalized value changes more drastically, which can help with learning;
        # tanh is a commonly used normalization method that normalizes the spec based on the desired spec (goal) and the current spec value as well,
        # but it does not have the property that normd has.
        self.lookup_style = env_config.get("lookup_style", "normd")

        # thresholds for Skip on Fail, if the spec_diff (the aggregated difference between current specs and desired specs) is less than min_threshold, it will be considered as failing the TT corner,
        # and the reward will be calculated based on how much it fails the TT corner using tt_threshold and min_threshold;
        self.min_threshold = env_config.get("min_threshold", -6.0)

        # the reward will be scaled by tt_threshold when the design fails the TT corner, and the reward will be scaled by min_threshold when the design fails the full corner but passes the TT corner,
        # and it will be a positive reward (e.g. 30.0) when the design passes all corners, so that the agent can learn to first pass the TT corner, then improve the design to pass the full corners.
        self.tt_threshold = env_config.get("tt_threshold", -1.0)
        self.tt_threshold_2 = env_config.get("tt_threshold_2", -3.0)
        self.concat_all_specs = env_config.get("concat_all_specs", False)
        self.verbose = env_config.get("verbose", False)
        self.alpha = env_config.get("alpha", 0.1)

        # Sample goal from target range every episode
        self.online_goal = env_config.get("online_goal", False)

        # Pareto goal sampling frequency
        self.pareto_freq = env_config.get("pareto_freq", 0)
        self.n_warmup = env_config.get("n_warmup", 4)
        self.env_steps = 0
        with open(self.CIR_YAML, "r") as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)
        self.yaml_data = yaml_data

        # single goal
        if self.generalize == False:
            specs = yaml_data["target_specs"]
        else:  # multi goals
            if self.online_goal == False:
                load_specs_path = (
                    ngspice_env.path + "/PPAAS/gen_specs/" + self.spec_path
                )
                with open(load_specs_path, "rb") as f:
                    specs = pickle.load(f)
            else:
                specs = yaml_data["target_specs"]
        self.specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
        # print("Specs used for env: ", self.specs)
        # print(type(self.specs))
        # ~~~
        # Specs used for env:  OrderedDict([('gain_min', (200, 300)), ('ibias_max', (0.001, 0.01)), ('phm_min', (60, 60.0000001)), ('t_settle_max', (2e-06, 1e-05)), ('ugbw_min', (1000000.0, 20000000.0)), ('vswing_min', (0.2, 0.3))])
        # <class 'collections.OrderedDict'>

        self.specs_ideal = []

        # specs_id is the list of spec names, e.g. ['gain_min', 'ibias_max', 'phm_min', 't_settle_max', 'ugbw_min', 'vswing_min']
        self.specs_id = list(self.specs.keys())
        self.num_os = len(list(self.specs.values())[0])
        self.cur_steps = 0
        self.full_sim = 0
        self.tt_sim = 0
        self.horizon = self.episode_len
        self.num_corners = len(yaml_data["dsn_netlist"])

        # param array
        params = yaml_data["params"]
        self.params_id = list(params.keys())
        self.params_val = list(params.values())

        # goal history buffer
        self.pareto_goal_history = []
        self.rejection = 0
        self.episode_steps = 0

        specs_range_dict = OrderedDict(
            sorted(self.yaml_data["target_specs"].items(), key=lambda k: k[0])
        )
        self.specs_range = list(specs_range_dict.values())
        self.episode_corner_norm_std = 1.0

        # initialize sim environment
        tt_design_netlist = [yaml_data["dsn_netlist"][0]]
        corner_design_netlists = yaml_data["dsn_netlist"][1:]
        self.corner_sim_env = CircuitClass(
            yaml_path=self.CIR_YAML,
            path=ngspice_env.path,
            design_netlists=corner_design_netlists,
        )
        self.tt_sim_env = CircuitClass(
            yaml_path=self.CIR_YAML,
            path=ngspice_env.path,
            design_netlists=tt_design_netlist,
        )
        self.full_sim_env = CircuitClass(yaml_path=self.CIR_YAML, path=ngspice_env.path)

        if env_config.get("action_type") == "discrete":
            self.action_meaning = [-1, 0, 2]
            self.action_space = spaces.MultiDiscrete(
                [len(self.action_meaning)] * len(self.params_id)
            )
        elif env_config.get("action_type") == "continuous":
            self.action_space = spaces.Box(
                low=np.array([ngspice_env.ACT_LOW] * len(self.params_id)),
                high=np.array([ngspice_env.ACT_HIGH] * len(self.params_id)),
            )

        if self.concat_all_specs:
            self.observation_space = spaces.Box(
                low=np.array(
                    [-np.inf] * (self.num_corners + 1) * len(self.specs_id)
                    + len(self.params_id) * [-np.inf]
                ),
                high=np.array(
                    [np.inf] * (self.num_corners + 1) * len(self.specs_id)
                    + len(self.params_id) * [np.inf]
                ),
                dtype=np.float64,
            )
            self.cur_specs = np.zeros(
                len(self.specs_id) * self.num_corners, dtype=np.float64
            )
        else:
            self.observation_space = spaces.Box(
                low=np.array(
                    [-np.inf] * 2 * len(self.specs_id) + len(self.params_id) * [-np.inf]
                ),
                high=np.array(
                    [np.inf] * 2 * len(self.specs_id) + len(self.params_id) * [np.inf]
                ),
                dtype=np.float64,
            )
            self.cur_specs = np.zeros(len(self.specs_id), dtype=np.float64)

        # initialize current parameters, parameters have diferent types of values,
        # some are discrete with different step sizes, some are continuous,
        # we will translate them into a unified parameter vector for the agent, and translate them back when simulating
        if env_config.get("action_type") == "discrete":
            self.cur_params = np.zeros(len(self.params_id), dtype=np.int32)
        elif env_config.get("action_type") == "continuous":
            self.cur_params = np.zeros(len(self.params_id), dtype=np.float64)

        # Get the g* (overall design spec) you want to reach
        self.g_star = np.array(yaml_data["normalize"])
        self.global_g = np.array(yaml_data["normalize"])

        # objective number (used for exploitation)
        self.obj_idx = 0

    def reset(self, seed=None, options=None):
        # if multi-goal is selected, every time reset occurs, it will select a different design spec as objective
        if self.generalize == True:
            if self.online_goal:
                # Only perform PGDS if history has sufficient data
                # PGDS stands for Pareto Guided Design Space Sampling,
                # which samples goals based on the pareto front of the goal history buffer, and the sampling frequency is determined by pareto_freq
                if (
                    len(self.pareto_goal_history) >= self.n_warmup
                    and self.pareto_freq > 0
                ):
                    if self.episode_steps % self.pareto_freq == 0:

                        # self.goal_idx is set via set_goal_idx function, which is called by the training script to loop through different goals in the candidate pool for better coverage,
                        # instead of randomly sampling from the candidate pool, which may lead to overfitting to certain goals and not covering the goal space well.
                        self.specs_ideal = self.specs_ideal_candidates[self.goal_idx]
                    else:
                        self.specs_ideal = self.sample_goal_uniform(
                            list(self.specs.values()), num_goals_to_sample=1
                        )[0]
                    self.episode_steps += 1
                else:
                    # Initially random sampling
                    self.specs_ideal = self.sample_goal_uniform(
                        list(self.specs.values()), num_goals_to_sample=1
                    )[0]
            else:
                if self.valid == True:
                    if self.obj_idx > self.num_os - 1:
                        self.obj_idx = 0
                    idx = self.obj_idx
                    self.obj_idx += 1
                else:
                    idx = random.randint(0, self.num_os - 1)
                self.specs_ideal = []
                for spec in list(self.specs.values()):
                    self.specs_ideal.append(spec[idx])
                self.specs_ideal = np.array(self.specs_ideal)
        else:
            self.specs_ideal = (
                self.g_star
            )  # (yaml ["normalize"] is the overall design spec g* you want to reach, which is used for single-goal setting)

        # only used for logging and analysis, not involved in reward calculation.
        # i.e., constructing the generated/updated obs.
        self.specs_ideal_norm = self.lookup(self.specs_ideal, self.global_g)

        self.cur_steps = 0
        self.cur_params = self.init_params()
        self.full_sim = 0
        self.tt_sim = 0
        tt_done = False
        min_threshold = self.min_threshold
        tt_threshold = self.tt_threshold

        # if Skip on Fail is selected, it will first simulate the design on the typical corner (TT corner), if the design does not meet the minimum threshold on the TT corner,
        # it will skip the full corner simulation and return a reward based on how much it fails the TT corner,
        # otherwise it will go to simulate the full corners and calculate reward based on all corners.
        # This can help with improving sample efficiency by skipping bad designs early, but it may also lead to suboptimal designs if the TT corner is not representative enough.
        if self.SoF:
            # 1st stage: TT corner simulation
            self.cur_specs = self.update(self.cur_params, self.tt_sim_env)[0]
            cur_spec_norms = []
            cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
            all_specs = np.array([self.cur_specs] * self.num_corners)
            spec_diff = self.aggregate(self.cur_specs, self.specs_ideal)
            if spec_diff < 0:
                reward = tt_threshold + (tt_threshold - min_threshold) * (
                    spec_diff / float(len(self.specs_id))
                )
            # 2nd stage: full corner simulation
            else:
                tt_done = True

                # simulate the full corners and get the specs for all corners, then calculate the reward based on the worst corner (the corner that has the lowest aggregated spec difference among all corners),
                # and also calculate the deviation among corners to encourage the agent to find more robust designs that can perform well across all corners instead of overfitting to a specific corner.
                full_cur_specs = self.update(self.cur_params, self.corner_sim_env)
                for cur_spec in full_cur_specs:
                    cur_spec_norms.append(self.lookup(cur_spec, self.global_g))
                cur_spec_norms = np.array([cur_spec_norm] + cur_spec_norms)
                reverse_indices = []
                for i in range(len(self.specs_id)):
                    if self.specs_id[i][-3:] == "max":
                        cur_spec_norms[:, i] = cur_spec_norms[:, i] * -1.0
                        reverse_indices.append(i)
                cur_spec_norm = np.min(cur_spec_norms, axis=0)
                worst_idx = np.argmin(cur_spec_norms, axis=0)
                reverse_indices = np.array(reverse_indices)
                cur_spec_norm[reverse_indices] = -cur_spec_norm[
                    reverse_indices
                ]  # not involved in reward calculation but used for logging and analysis
                self.cur_specs = np.array(
                    [self.cur_specs] + full_cur_specs
                )  # not involved in reward calculation but used for logging and analysis
                all_specs = copy.deepcopy(
                    self.cur_specs
                )  # not involved in reward calculation but used for logging and analysis
                self.cur_specs = self.cur_specs[worst_idx, np.arange(len(worst_idx))]
                spec_diff = self.aggregate(self.cur_specs, self.specs_ideal)
                if spec_diff < 0:
                    reward = -tt_threshold * spec_diff / float(len(self.specs_id))

                # 3rd stage: satified all desired specs
                else:
                    reward = 30.0

        else:
            tt_done = False
            self.cur_specs = self.update(self.cur_params, self.full_sim_env)
            cur_spec_norms = []
            for cur_spec in self.cur_specs:
                cur_spec_norms.append(self.lookup(cur_spec, self.global_g))
            cur_spec_norms = np.array(cur_spec_norms)
            reverse_indices = []
            for i in range(len(self.specs_id)):
                # specs_id is the list of spec names, e.g. ['gain_min', 'ibias_max', 'phm_min', 't_settle_max', 'ugbw_min', 'vswing_min'],
                #  if the spec is a max spec, then we need to reverse the sign of the normalized spec value for reward calculation,
                # because for max spec, when the current spec value is higher than the desired spec value, it should be considered as better and have a higher reward, which is the opposite for min spec.
                if self.specs_id[i][-3:] == "max":
                    cur_spec_norms[:, i] = cur_spec_norms[:, i] * -1.0
                    reverse_indices.append(i)

            cur_spec_norm = np.min(cur_spec_norms, axis=0)
            worst_idx = np.argmin(cur_spec_norms, axis=0)
            reverse_indices = np.array(reverse_indices)
            cur_spec_norm[reverse_indices] = -cur_spec_norm[reverse_indices]
            self.cur_specs = np.array(self.cur_specs)
            self.cur_specs = self.cur_specs[worst_idx, np.arange(len(worst_idx))]
            spec_diff = self.aggregate(self.cur_specs, self.specs_ideal)
            if self.concat_all_specs:
                cur_spec_norm = (
                    cur_spec_norms.flatten()
                )  # not involved in reward calculation but used for logging and analysis
            if spec_diff < 0:
                reward = -min_threshold * spec_diff / float(len(self.specs_id))
            else:
                reward = 30.0

        if reward >= 0:
            done = True
        else:
            done = False
        truncated = False

        info = {
            "reward": reward,
            "params": self.cur_params,
            "target_specs": self.specs_ideal,
            "cur_specs": self.cur_specs,
            "all_specs": all_specs,
            "done": done,
            "truncated": truncated,
            "tt_done": tt_done,
            "pareto_buffer_size": len(self.pareto_goal_history),
        }
        self.ob = np.concatenate(
            [cur_spec_norm, self.specs_ideal_norm, self.cur_params]
        )
        return self.ob, info

    def step(self, action):
        action = list(np.reshape(np.array(action), (np.array(action).shape[0],)))
        self.cur_params = self.update_params(action)
        tt_done = False
        worst_idx = [0] * len(self.specs_id)
        min_threshold = self.min_threshold
        tt_threshold = self.tt_threshold

        if self.SoF:
            # 1st stage: TT corner simulation
            self.cur_specs = self.update(self.cur_params, self.tt_sim_env)[0]
            cur_spec_norms = []
            cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
            all_specs = np.array([self.cur_specs] * self.num_corners)
            spec_diff = self.aggregate(self.cur_specs, self.specs_ideal)
            if spec_diff < 0:
                tt_done = False
                self.tt_sim += 1

                # reward is based on how much it fails the TT corner,
                #  and it is scaled by the number of specs to keep the reward magnitude consistent across different environments with different number of specs
                reward = tt_threshold + (tt_threshold - min_threshold) * (
                    spec_diff / float(len(self.specs_id))
                )
                corner_norm_std = 1.0
            else:
                # 2nd stage: full corner simulation
                tt_done = True
                self.full_sim += 1
                full_cur_specs = self.update(self.cur_params, self.corner_sim_env)
                for cur_spec in full_cur_specs:
                    cur_spec_norms.append(self.lookup(cur_spec, self.global_g))
                cur_spec_norms = np.array([cur_spec_norm] + cur_spec_norms)
                reverse_indices = []
                for i in range(len(self.specs_id)):
                    if self.specs_id[i][-3:] == "max":
                        cur_spec_norms[:, i] = cur_spec_norms[:, i] * -1.0
                        reverse_indices.append(i)
                cur_spec_norm = np.min(cur_spec_norms, axis=0)
                worst_idx = np.argmin(cur_spec_norms, axis=0)
                reverse_indices = np.array(reverse_indices)
                cur_spec_norm[reverse_indices] = -cur_spec_norm[reverse_indices]
                cur_spec_norms[:, reverse_indices] = -cur_spec_norms[:, reverse_indices]
                self.cur_specs = np.array([self.cur_specs] + full_cur_specs)
                all_specs = copy.deepcopy(self.cur_specs)
                self.cur_specs = self.cur_specs[worst_idx, np.arange(len(worst_idx))]
                spec_diff = self.aggregate(self.cur_specs, self.specs_ideal)

                # corner_norm_std is the standard deviation of the normalized spec values among different corners, which can be used as a measure of the robustness of the design across different corners,
                # and it can be used to encourage the agent to find more robust designs that can perform well across all corners instead of overfitting to a specific corner.
                corner_norm_std = np.sqrt(
                    np.mean((all_specs[1:] / all_specs[0] - 1) ** 2)
                )
                corner_norm_std = np.clip(corner_norm_std, 0, 1)
                if spec_diff < 0:
                    reward = -tt_threshold * spec_diff / float(len(self.specs_id))
                else:
                    reward = 30.0

        else:
            self.full_sim += 1
            self.cur_specs = self.update(self.cur_params, self.full_sim_env)
            cur_spec_norms = []
            for cur_spec in self.cur_specs:
                cur_spec_norms.append(self.lookup(cur_spec, self.global_g))
            cur_spec_norms = np.array(cur_spec_norms)
            reverse_indices = []
            for i in range(len(self.specs_id)):
                if self.specs_id[i][-3:] == "max":
                    cur_spec_norms[:, i] = cur_spec_norms[:, i] * -1.0
                    reverse_indices.append(i)

            cur_spec_norm = np.min(cur_spec_norms, axis=0)
            worst_idx = np.argmin(cur_spec_norms, axis=0)
            reverse_indices = np.array(reverse_indices)
            cur_spec_norm[reverse_indices] = -cur_spec_norm[reverse_indices]
            self.cur_specs = np.array(self.cur_specs)
            all_specs = copy.deepcopy(self.cur_specs)
            self.cur_specs = self.cur_specs[worst_idx, np.arange(len(worst_idx))]
            spec_diff = self.aggregate(self.cur_specs, self.specs_ideal)
            corner_norm_std = np.sqrt(np.mean((all_specs[1:] / all_specs[0] - 1) ** 2))
            corner_norm_std = np.clip(corner_norm_std, 0, 1)
            if self.concat_all_specs:
                cur_spec_norm = cur_spec_norms.flatten()
            if spec_diff < 0:
                reward = -min_threshold * spec_diff / float(len(self.specs_id))
            else:
                reward = 30.0

        # add deviation penalty
        reward = reward - self.alpha * corner_norm_std

        self.env_steps = self.env_steps + 1
        self.cur_steps = self.cur_steps + 1

        if reward >= 0:
            done = True
        else:
            done = False

        if self.cur_steps >= self.horizon:
            truncated = True
        else:
            truncated = False

        if done and self.online_goal:
            self.pareto_goal_history = self.update_pareto_goals(
                self.specs_ideal, self.pareto_goal_history
            )

        if done or truncated:
            self.episode_corner_norm_std = corner_norm_std
            if self.online_goal:
                if self.pareto_freq > 0:

                    # self.specs_ideal_candidates is the candidate pool of goals for online goal sampling, which is updated every time an episode ends, and if the episode ends with a good design (reward >= 0), then the design spec of that episode will be added to the pareto goal history buffer,
                    # see:
                    # self.pareto_goal_history = self.update_pareto_goals(
                    #     self.specs_ideal, self.pareto_goal_history
                    # )

                    self.specs_ideal_candidates = self.sample_goal_pareto(
                        self.pareto_goal_history, num_goals_to_sample=16
                    )
                else:
                    self.specs_ideal_candidates = self.sample_goal_uniform(
                        list(self.specs.values()), num_goals_to_sample=16
                    )
                self.goals_norm = self.lookup(
                    self.specs_ideal_candidates, self.global_g
                )

        self.ob = np.concatenate(
            [cur_spec_norm, self.specs_ideal_norm, self.cur_params]
        )
        info = {
            "reward": reward,
            "params": self.cur_params,
            "target_specs": self.specs_ideal,
            "cur_specs": self.cur_specs,
            "all_specs": all_specs,
            "worst_index": worst_idx,
            "done": done,
            "tt_done": tt_done,
            "truncated": truncated,
            "full_sim": self.full_sim,
            "tt_sim": self.tt_sim,
            "ep_corner_norm_std": self.episode_corner_norm_std,
            "pareto_buffer_size": len(self.pareto_goal_history),
        }
        return self.ob, reward, done, truncated, info

    def lookup(self, spec, goal_spec):
        """Normalize the spec based on the goal_spec using the specified lookup style."""
        spec = np.asarray(spec, dtype=np.float32)
        goal_spec = np.asarray(goal_spec, dtype=np.float32)
        epsilon = 1e-9
        goal_spec = np.where(goal_spec == 0, epsilon, goal_spec)
        if self.lookup_style == "normd":
            delta = spec - goal_spec
            abs_delta = np.abs(delta) + epsilon  # Avoid log(0)
            denom = goal_spec + np.abs(spec) + epsilon  # Avoid log(0)
            norm_spec = np.sign(delta) * np.exp(np.log(abs_delta) - np.log(denom))

        elif self.lookup_style == "tanh":
            try:
                scale_factor = 10.0
                norm_spec = np.tanh(
                    (spec - goal_spec) / (scale_factor * goal_spec)
                ) / np.tanh(1 / scale_factor)
            except RuntimeWarning as e:
                pdb.set_trace()
        return norm_spec

    def unlookup(self, norm_spec, goal_spec):
        if self.lookup_style == "normd":
            spec = -1 * np.multiply((norm_spec + 1), goal_spec) / (norm_spec - 1)
        elif self.lookup_style == "tanh":
            try:
                scale_factor = 10.0
                x = np.clip(norm_spec * np.tanh(1 / scale_factor), -0.999, 0.999)
                spec = goal_spec * (1 + scale_factor * np.arctanh(x))
            except RuntimeWarning as e:
                pdb.set_trace()
        return spec

    def aggregate(self, spec, goal_spec):
        """Aggregate the spec values into a single scalar for reward calculation. Here we use a simple sum of normalized spec differences, but other aggregation methods can be used."""
        rel_specs = self.lookup(spec, goal_spec)
        pos_val = []
        reward = 0.0
        for i, rel_spec in enumerate(rel_specs):
            if self.specs_id[i][-3:] == "max":
                rel_spec = rel_spec * -1.0
            if rel_spec < 0:
                reward += rel_spec
                pos_val.append(0)
            else:
                pos_val.append(1)
        return reward if reward < -0.02 else 10.0

    def update(self, cur_params: list[Union[int, float]], sim_env):
        """Translate the current parameters into actual parameter values for simulation, run the simulation, and return the current specs."""
        params = self.translate_params(cur_params)
        param_val = [OrderedDict(list(zip(self.params_id, params)))]
        states, specs, infos = sim_env.run(param_val[0])
        cur_specs = []
        for spec in specs:
            cur_spec = OrderedDict(sorted(spec.items(), key=lambda k: k[0]))
            cur_specs.append(np.array(list(cur_spec.values())))
        return cur_specs

    def set_goal_idx(self, goal_idx):
        self.goal_idx = goal_idx
        return True

    def sample_goal_pareto(self, pareto_goals, num_goals_to_sample=1):
        sampled_goals = []
        while len(sampled_goals) < num_goals_to_sample:
            goal = self.sample_goal_uniform(self.specs_range, num_goals_to_sample=1)[0]
            dominated, dominant = self.is_goal_dominated(goal, pareto_goals)
            if not dominated:
                sampled_goals.append(goal)

                # self.rejection is the number of consecutive times a sampled goal is rejected for being dominated by the pareto front, and if it exceeds a certain threshold (e.g. 100),
                # we will stop using pareto sampling and switch to uniform sampling to avoid getting stuck in a local region of the goal space.
                self.rejection = 0
            else:
                self.rejection += 1
        return np.array(sampled_goals)

    def sample_goal_uniform(self, specs_range_vals, num_goals_to_sample=1):
        """Uniformly sample goals from the specified ranges for each spec."""

        sampled_goals = []
        while len(sampled_goals) < num_goals_to_sample:
            specs_valid = []
            for spec in specs_range_vals:
                # check spec is discrete or continuous by checking its type (int or float)
                if isinstance(spec[0], int):
                    specs_valid.append(random.randint(int(spec[0]), int(spec[1])))
                else:
                    specs_valid.append(random.uniform(float(spec[0]), float(spec[1])))
            sampled_goals.append(specs_valid)
        return np.array(sampled_goals)

    def update_pareto_goals(self, new_goal, pareto_goals):
        dominated, _ = self.is_goal_dominated(new_goal, pareto_goals)
        if dominated:
            return pareto_goals
        # Remove any existing goals that the new goal dominates
        pareto_goals = [
            goal for goal in pareto_goals if not self.is_dominated(goal, new_goal)
        ]
        pareto_goals.append(new_goal)
        return pareto_goals

    def is_goal_dominated(self, goal, goal_set):
        """Check if the goal is dominated by any goal in the set."""
        for competitor in goal_set:
            if self.is_dominated(goal, competitor):
                return True, competitor
        return False, None

    def is_dominated(self, p, q):
        """Check if suggested goal p is dominated by goal q (assuming minimization or maximization)."""
        for idx in range(len(self.specs_id)):
            if (
                self.specs_id[idx] == "phm_min"
            ):  # phm_min's desired spec range is a singleton, not a range
                continue
            if self.specs_id[idx][-3:] == "max":
                if p[idx] < q[idx]:
                    return False
            else:
                if p[idx] > q[idx]:
                    return False
        return True

    def reset_idx(self):
        self.obj_idx = 0

    def get_goals_norm(self):
        return self.goals_norm

    def get_specs_ideal(self):
        return self.specs_ideal

    def get_specs_ideal_norm(self):
        return self.specs_ideal_norm

    @abstractmethod
    def init_params(self):
        pass

    @abstractmethod
    def update_params(self, action):
        pass

    @abstractmethod
    def translate_params(self, cur_params):
        pass
