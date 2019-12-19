# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objects as go
import plotly.express as px

from ... import Memory
from ... import Hyperactive


class Insight:
    def __init__(self, search_config, X, y):
        self.search_config = search_config
        self.X = X
        self.y = y

    def plot_performance(self, runs=3, path=None, optimizers="all"):
        if optimizers == "all":
            optimizers = [
                "HillClimbing",
                "StochasticHillClimbing",
                "TabuSearch",
                "RandomSearch",
                "RandomRestartHillClimbing",
                "RandomAnnealing",
                "SimulatedAnnealing",
                "StochasticTunneling",
                "ParallelTempering",
                "ParticleSwarm",
                "EvolutionStrategy",
                "Bayesian",
            ]

        eval_times = []
        total_times = []
        for run in range(runs):

            eval_time = []
            total_time = []
            for optimizer in optimizers:
                opt = Hyperactive(self.X, self.y, memory=False)
                opt.search(self.search_config, n_iter=3, optimizer=optimizer)

                eval_time.append(opt.eval_time)
                total_time.append(opt.total_time)

            eval_time = np.array(eval_time)
            total_time = np.array(total_time)

            eval_times.append(eval_time)
            total_times.append(total_time)

        eval_times = np.array(eval_times)
        total_times = np.array(total_times)
        opt_times = np.subtract(total_times, eval_times)

        opt_time_mean = opt_times.mean(axis=0)
        eval_time_mean = eval_times.mean(axis=0)
        total_time_mean = total_times.mean(axis=0)

        # opt_time_std = opt_times.std(axis=0)
        # eval_time_std = eval_times.std(axis=0)

        eval_time = eval_time_mean / total_time_mean
        opt_time = opt_time_mean / total_time_mean

        fig = go.Figure(
            data=[
                go.Bar(name="Eval time", x=optimizers, y=eval_time),
                go.Bar(name="Opt time", x=optimizers, y=opt_time),
            ]
        )
        fig.update_layout(barmode="stack")
        py.offline.plot(fig, filename="sampleplot.html")

    def plot_search_path(self, path=None, optimizers=["HillClimbing"]):
        for optimizer in optimizers:
            opt = Hyperactive(self.X, self.y, memory=False, verbosity=10)
            opt.search(self.search_config, n_iter=20, optimizer=optimizer)

            pos_list = opt.pos_list
            score_list = opt.score_list

            pos_list = np.array(pos_list)
            score_list = np.array(score_list)

            pos_list = np.squeeze(pos_list)
            score_list = np.squeeze(score_list)

            df = pd.DataFrame(
                {
                    "n_neighbors": pos_list[:, 0],
                    "leaf_size": pos_list[:, 1],
                    "score": score_list,
                }
            )

            layout = go.Layout(xaxis=dict(range=[0, 50]), yaxis=dict(range=[0, 50]))
            fig = go.Figure(
                data=go.Scatter(
                    x=df["n_neighbors"],
                    y=df["leaf_size"],
                    mode="lines+markers",
                    marker=dict(
                        size=10,
                        color=df["score"],
                        colorscale="Viridis",  # one of plotly colorscales
                        showscale=True,
                    ),
                ),
                layout=layout,
            )

            py.offline.plot(fig, filename="search_path" + optimizer + ".html")
