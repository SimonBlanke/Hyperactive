from .adapter._hyper_gradient_conv import HyperGradientConv


class Runner:
    def __init__(self, optimizer_class, opt_params):
        self.optimizer_class = optimizer_class
        self.opt_params = opt_params

    def run_search(self, search_info, nth_process, max_time, p_bar):
        self.search_info = search_info

        self.hg_conv = HyperGradientConv(search_info.s_space)

        search_space_positions = search_info.s_space.positions
        initialize = self.hg_conv.conv_initialize(search_info.initialize)

        # conv warm start for smbo from values into positions
        if "warm_start_smbo" in self.opt_params:
            self.opt_params["warm_start_smbo"] = self.hg_conv.conv_memory_warm_start(
                self.opt_params["warm_start_smbo"]
            )

        gfo_constraints = [
            Constraint(constraint, self.s_space)
            for constraint in search_info.constraints
        ]

        self.gfo_optimizer = self.optimizer_class(
            search_space=search_space_positions,
            initialize=initialize,
            constraints=gfo_constraints,
            random_state=search_info.random_state,
            nth_process=nth_process,
            **self.opt_params,
        )

        self.gfo_optimizer.init_search(
            search_info.experiment.gfo_objective_function,
            search_info.n_iter,
            max_time,
            search_info.max_score,
            search_info.early_stopping,
            search_info.memory,
            search_info.memory_warm_start,
            False,
        )
        for nth_iter in range(search_info.n_iter):
            if p_bar:
                p_bar.set_description(
                    "["
                    + str(nth_process)
                    + "] "
                    + str(search_info.experiment.__class__.__name__)
                    + " ("
                    + self.optimizer_class.name
                    + ")",
                )

            self.gfo_optimizer.search_step(nth_iter)
            if self.gfo_optimizer.stop.check():
                break

            if p_bar:
                p_bar.set_postfix(
                    best_score=str(self.gfo_optimizer.score_best),
                    best_pos=str(self.gfo_optimizer.pos_best),
                    best_iter=str(self.gfo_optimizer.p_bar._best_since_iter),
                )

                p_bar.update(1)
                p_bar.refresh()

        self.gfo_optimizer.finish_search()

    def convert_results2hyper(self):
        self.eval_times = sum(self.gfo_optimizer.eval_times)
        self.iter_times = sum(self.gfo_optimizer.iter_times)

        if self.gfo_optimizer.best_para is not None:
            value = self.hg_conv.para2value(self.gfo_optimizer.best_para)
            position = self.hg_conv.position2value(value)
            best_para = self.hg_conv.value2para(position)
            self.best_para = best_para
        else:
            self.best_para = None

        self.best_score = self.gfo_optimizer.best_score
        self.positions = self.gfo_optimizer.search_data
        self.search_data = self.hg_conv.positions2results(self.positions)

        results_dd = self.gfo_optimizer.search_data.drop_duplicates(
            subset=self.search_info.s_space.dim_keys, keep="first"
        )
        self.memory_values_df = results_dd[
            self.search_info.s_space.dim_keys + ["score"]
        ].reset_index(drop=True)
