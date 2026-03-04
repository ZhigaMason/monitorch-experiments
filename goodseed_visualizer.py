import goodseed
import numpy as np
from collections import OrderedDict
from monitorch.visualizer import AbstractVisualizer, PlayerVisualizer, PrintVisualizer
from typing import Any


class GoodSeedVisualizer(AbstractVisualizer):
    def __init__(
        self,
        *,
        hp_dicts: dict[str, Any] | None = None,
        prefix: str = "train/monitorch",
        **kwargs,
    ):
        self.run = goodseed.Run(**kwargs)
        self.prefix = prefix
        print(hp_dicts)
        hp_dicts = hp_dicts if hp_dicts is not None else {}
        print(hp_dicts)
        for name, value in hp_dicts.items():
            self.run[name] = value

    def close(self):
        self.run.close()

    def __del__(self):
        self.close()

    def plot_numerical_values(
        self,
        epoch: int,
        main_tag: str,
        values_dict: OrderedDict[str, dict[str, float]],
        ranges_dict: OrderedDict[str, dict[tuple[str, str], tuple[float, float]]]
        | None = None,
    ) -> None:
        main_prefix = self.prefix + "/" + main_tag

        for value_name, value_dict in values_dict.items():
            value_prefix = main_prefix + "/" + value_name
            for stat_name, stat in value_dict.items():
                stat = (
                    stat if np.isfinite(stat) else 0
                )  # guard against NaNs, see notes4matej.md
                self.run[value_prefix + "/" + stat_name].log(stat, step=epoch)

        for range_name, range_dict in ranges_dict.items():
            range_prefix = main_prefix + "/" + range_name
            for stat_name, stat in range_dict.items():
                self.run[range_prefix + "/" + stat_name[0]].log(
                    stat[0] if np.isfinite(stat[0]) else 0, step=epoch
                )
                self.run[range_prefix + "/" + stat_name[1]].log(
                    stat[1] if np.isfinite(stat[1]) else 0, step=epoch
                )

    def plot_probabilities(
        self,
        epoch: int,
        main_tag: str,
        values_dict: OrderedDict[str, dict[str, float]],
    ) -> None:
        main_prefix = self.prefix + "/" + main_tag

        for value_name, value_dict in values_dict.items():
            value_prefix = main_prefix + "/" + value_name
            for stat_name, stat in value_dict.items():
                stat = stat if np.isfinite(stat) else 0
                self.run[value_prefix + "/" + stat_name].log(stat, step=epoch)

    def plot_relations(
        self,
        epoch: int,
        main_tag,
        values_dict: OrderedDict[str, dict[str, float]],
    ) -> None:
        # not implementing relations as they are extremely ugly without stack plot
        pass

    def register_tags(self, main_tag: str, tag_attr) -> None:
        """
        Prepare visualizer's inner state for plot.

        Parameters
        ----------
        main_tag : str
            Name of the collection of plots.
        tag_attr : TagAttributes
            Tag attributes to configure plot.
        """
        pass


if __name__ == "__main__":
    # Class for GoodSeed visualization
    vis = GoodSeedVisualizer(
        hp_dicts={"moments": "default", "lr": 3e-2},
        run_id="OPT-03",
    )
    # Class for debug on monitorch side

    # vis = PrintVisualizer()
    player = PlayerVisualizer(
        "experiments/adamw_basic_showcase/logs/adam_lr3e-2_wd0.05_moments-default.pkl",
        vis,
    ).playback()
