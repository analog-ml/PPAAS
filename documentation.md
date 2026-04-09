In ngspice_env_base.py,  `self.specs_ideal` variable stores the current selected/chosen spec for training
    - Newly selected value of `self.specs_ideal` is configured in `reset()` method (ngspice_env_base.py)
    - In the `step()`, possible candidates for later selection of `self.specs_ideal` are saved in `self.specs_ideal_candidates`
    - `self.goal_idx` (with a setter: `self.set_goal_idx(.)`) serves as the imtermediate variable for selecting the `self.specs_ideal` from the candidate list `self.specs_ideal_candidates`
    - The `self.goal_idx` is set via `self.training_env.env_method("set_goal_idx", sampled_goal_idx)` in train.py
        - This happens when we have the FLAGs: `done` or `truncated` set to True
        - The `goal_idx` is obtained via model prediction and sampling (Line~131, train.py)