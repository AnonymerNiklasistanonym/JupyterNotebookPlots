import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import FuncFormatter
from fractions import Fraction

from typing import Callable, NotRequired, Optional, Union, TypedDict, Unpack


class GenericPlotConfigurationParams(TypedDict):
    title: str
    y_label: str
    x_label: str
    limits_x: NotRequired[tuple[Union[float, None], Union[float, None]]]
    limits_y: NotRequired[tuple[Union[float, None], Union[float, None]]]
    integer_x_ticks: NotRequired[bool]
    custom_x_ticks: NotRequired[list[float]]
    custom_y_ticks: NotRequired[list[float]]
    fraction_x_ticks: NotRequired[bool]
    fraction_y_ticks: NotRequired[bool]
    draw_grid: NotRequired[bool]


class GenericPlotShowExportParams(TypedDict):
    draw_legend: NotRequired[str]
    output_file_path: NotRequired[str]
    show: NotRequired[bool]
    """True is the default"""


class GenericPlotParams(GenericPlotConfigurationParams, GenericPlotShowExportParams):
    pass


def remove_duplicates_list[t](generic_list: list[t]) -> list[t]:
    new_list: list[t] = []
    for generic_list_element in generic_list:
        if generic_list_element not in new_list:
            if isinstance(generic_list_element, Fraction):
                new_list.append(float(generic_list_element))
            else:
                new_list.append(generic_list_element)
    return new_list


def _plot_generic_configuration(plot: plt, **kwargs: Unpack[GenericPlotParams]):
    """
    Generic configuration of a plot (using generic plot kwargs).
    """
    plot.figure()
    # > Set labels and title
    plot.title(kwargs["title"])
    plot.ylabel(kwargs["y_label"])
    plot.xlabel(kwargs["x_label"])
    # > Optionally set plot draw limits
    if kwargs.get("limits_x", None):
        plot.gca().set_xlim(kwargs["limits_x"][0], kwargs["limits_x"][1])
    if kwargs.get("limits_y", None):
        plot.gca().set_ylim(kwargs["limits_y"][0], kwargs["limits_y"][1])
    # > Optionally only render integer x ticks
    if kwargs.get("integer_x_ticks", False):
        plot.gca().xaxis.get_major_locator().set_params(integer=True)
    # > Optionally set custom axis ticks
    if kwargs.get("custom_x_ticks", None):
        plot.xticks(remove_duplicates_list(kwargs["custom_x_ticks"]))
    if kwargs.get("custom_y_ticks", None):
        plot.yticks(remove_duplicates_list(kwargs["custom_y_ticks"]))
    # > Optionally use fractions for axis ticks
    if kwargs.get("fraction_x_ticks", False):
        plot.gca().xaxis.set_major_formatter(
            FuncFormatter(lambda x, y: str(Fraction(x).limit_denominator()))
        )
    if kwargs.get("fraction_y_ticks", False):
        plot.gca().yaxis.set_major_formatter(
            FuncFormatter(lambda x, y: str(Fraction(x).limit_denominator()))
        )
    # > Optionally draw a grid
    if kwargs.get("draw_grid", False):
        # Keep the grid behind the bars
        plot.gca().set_axisbelow(True)
        plot.grid(True)


def _plot_generic_show_export(plot: plt, **kwargs: Unpack[GenericPlotShowExportParams]):
    """
    Export/Show a plot (using generic plot kwargs).
    """
    # > Optionally add a legend (now that all data was plotted)
    if kwargs.get("draw_legend", None):
        # Add legend
        plot.legend(loc=kwargs["draw_legend"])
    # Optionally save the plot in a file:
    if kwargs.get("output_file_path", None):
        plot.savefig(kwargs["output_file_path"])
    # Optionally draw the plot
    if kwargs.get("show", True):
        plot.show()


def plot_random_variable_distribution(
    data: dict[float, float],
    bar_color="blue",
    bar_width=0.1,
    data_col_x_name: str = None,
    data_col_p_x_eq_x_name: str = None,
    draw_only_data_x_ticks=False,
    draw_only_data_y_ticks=False,
    **kwargs: Unpack[GenericPlotParams],
) -> pd.DataFrame:
    """
    Draw a bar diagram that represents the distribution of a random variable.

    For a finite probability space (Omega = sample space, P = probability space)
    a random variable is defined as X : Omega -> R.

    | omega in Omega | omega_1    | ... | omega_n    |
    | -------------- | ---------- | --- | ---------- |
    | X(omega)       | X(omega_1) | ... | X(omega_n) |

    -> X(Omega) = { X(omega_1), ..., X(omega_n) } subset R

    A random variable has it's own probability space P^X : Power(Omega) -> R
    which is defined as P^X(A in X(Omega)) = P(X^-1(A)).
    Where X^-1(A) = { omega in Omega | X(omega) in A } meaning a set consisting
    of all sample space elements that become part of the set A when applying X.
    Additional definitions are that X^-1({ x }) := { X = x } and that
    P({ X = x }) can also be written as P(X = x).

    | x in X(Omega)                          | x_1               | ... | x_n |
    | -------------------------------------- | ----------------- | --- | --- |
    | X^-1({ x }) = { X = x }                | X^-1({ x_1 })     | ... | ... |
    | P^X({ x }) = P(X^-1({ x })) = P(X = x) | P(X^-1({ x_1 }))  | ... | ... |

    Parameters:
    data: The random variable data stored in a dict[x in X(Omega), P(X = x)]).

    Returns:
    Show graph and return the data as a table.
    """
    # Configure plot (required)
    kwargs.setdefault("title", "Verteilung von $X$")
    kwargs.setdefault("x_label", "$x \\in X(\\Omega)$")
    kwargs.setdefault("y_label", "$\\mathbb{P}(X = x)$")
    # Create data
    if data_col_x_name is None:
        data_col_x_name = kwargs["x_label"]
    if data_col_p_x_eq_x_name is None:
        data_col_p_x_eq_x_name = kwargs["y_label"]
    data_pd = pd.DataFrame(
        data={
            data_col_x_name: list(data.keys()),
            data_col_p_x_eq_x_name: list(data.values()),
        }
    )
    # Configure plot (optional)
    if draw_only_data_x_ticks:
        kwargs.setdefault("custom_x_ticks", [0, *data_pd[data_col_x_name]])
    if draw_only_data_y_ticks:
        kwargs.setdefault("custom_y_ticks", [0, *data_pd[data_col_p_x_eq_x_name]])
    kwargs.setdefault("integer_x_ticks", True)
    kwargs.setdefault("draw_fractions", True)
    kwargs.setdefault("draw_grid", True)
    _plot_generic_configuration(plot=plt, **kwargs)
    # Plot data
    plt.bar(
        data_pd[data_col_x_name],
        data_pd[data_col_p_x_eq_x_name],
        width=bar_width,
        color=bar_color,
    )
    # Show/export plot and return the data
    _plot_generic_show_export(plt, **kwargs)
    return data_pd.sort_values(by=[data_col_x_name], ascending=True)


def plot_random_variable_distribution_function(
    data: dict[int, float],
    data_col_x_name: str = None,
    data_col_p_x_eq_x_name: str = None,
    data_col_p_x_leq_x_name: str = None,
    draw_only_data_x_ticks=False,
    draw_only_data_y_ticks=False,
    line_color="blue",
    line_width=4,
    marker_size=8,
    **kwargs: Unpack[GenericPlotParams],
) -> pd.DataFrame:
    """
    Draw a specific graph that represents the distribution function of a random variable.

    (See description of plot_random_variable_distribution for basics)

    If you have the same input data as in plot_random_variable_distribution you
    can trivially compute P(X <= x) since this is just the sum of all x_i that
    are equal or less: P(X <= x_i) = P(X = x_1) + ... + P(X = x_i)
    (assuming x_i to x_n are sorted ascending)

    | x in X(Omega) | x_1                    | ... | x_n                    |
    | ------------- | ---------------------- | --- | ---------------------- |
    | P(X = x)      | P(X = x_1)             | ... | P(X = x_n)             |
    | P(X <= x)     | sum^1{i=1}(P(X = x_i)) | ... | sum^n{i=1}(P(X = x_i)) |

    Parameters:
    data: The random variable data stored in a dict[x in X(Omega), P(X = x)]).

    Returns:
    Show graph and return the data as a table.
    """
    # Configure plot (required)
    kwargs.setdefault("title", "Verteilungsfunktion $F^X$")
    kwargs.setdefault("x_label", "$x$")
    kwargs.setdefault("y_label", "$F^X(x) = \\mathbb{P}(X \\leq x)$")
    if data_col_x_name is None:
        data_col_x_name = kwargs["x_label"]
    if data_col_p_x_eq_x_name is None:
        data_col_p_x_eq_x_name = "$\\mathbb{P}(X = x)$"
    if data_col_p_x_leq_x_name is None:
        data_col_p_x_leq_x_name = kwargs["y_label"]
    # Create data
    data_pd = pd.DataFrame(
        data={
            data_col_x_name: list(data.keys()),
            data_col_p_x_eq_x_name: list(data.values()),
            data_col_p_x_leq_x_name: [
                sum(list(data.values())[0 : i + 1]) for i, _ in enumerate(data.values())
            ],
        }
    ).sort_values(by=[data_col_x_name], ascending=True)
    # Configure generic plot options
    max_x_value = max(data_pd[data_col_x_name])
    if draw_only_data_x_ticks:
        kwargs.setdefault(
            "custom_x_ticks",
            [-1, 0, *data_pd[data_col_x_name], round(max_x_value) + 1],
        )
    if draw_only_data_y_ticks:
        kwargs.setdefault(
            "custom_y_ticks",
            [0, *data_pd[data_col_p_x_leq_x_name]],
        )
    kwargs.setdefault("limits_x", (-1, max_x_value + 1))
    kwargs.setdefault("integer_x_ticks", True)
    kwargs.setdefault("draw_grid", True)
    _plot_generic_configuration(plt, **kwargs)
    # Plot data
    # > Draw initial line from -1 to 0
    plt.plot(
        [-1, 0], [0, 0], linewidth=line_width, color=line_color, solid_capstyle="butt"
    )
    # > Draw final line from max to max + 1
    plt.plot(
        [max_x_value, max_x_value + 1],
        [1, 1],
        linewidth=line_width,
        color=line_color,
        solid_capstyle="butt",
    )
    # > Draw the data points (rows must be sorted after x!)
    previous_pos = (0, 0)
    for index, row in data_pd.iterrows():
        current_pos = (row[data_col_x_name], row[data_col_p_x_leq_x_name])
        # The blue line
        plt.plot(
            [previous_pos[0], current_pos[0]],
            [previous_pos[1], previous_pos[1]],
            linewidth=line_width,
            color=line_color,
            solid_capstyle="butt",
        )
        # The blue circle
        if previous_pos[1] != current_pos[1]:
            plt.plot(
                [current_pos[0], current_pos[0]],
                [current_pos[1], current_pos[1]],
                "o",
                linewidth=0,
                color=line_color,
                markersize=marker_size,
            )
        previous_pos = current_pos
    # Show/export plot and return the data
    _plot_generic_show_export(plt, **kwargs)
    return data_pd


def plot_random_variable_constant_distribution_function(
    data_func: Callable[[float], float],
    data_range: tuple[float, float],
    data_samples=10000,
    line_color="red",
    line_width=2,
    data_col_x_name: str = None,
    data_col_p_x_leq_x_name: str = None,
    data_return=False,
    **kwargs: Unpack[GenericPlotParams],
) -> Optional[pd.DataFrame]:
    """
    Draw a specific graph that represents the distribution function of a random variable
    that is represented by a constant function.

    (See description of plot_random_variable_distribution_function for basics)

    Parameters:
    data_func: A function that returns the P(X <= x) value given x in X(Omega).
    data_range: The range of the function.
    data_samples: The amount of data points to be created in the defined range.

    Returns:
    Show graph and return the data as a table.
    """
    # Configure plot (required)
    kwargs.setdefault("title", "Konstante Verteilungsfunktion $F^X$")
    kwargs.setdefault("x_label", "$x \\in X(\\Omega)$")
    kwargs.setdefault("y_label", "$F^X(x) = \\mathbb{P}(X \\leq x)$")
    if data_col_x_name is None:
        data_col_x_name = kwargs["x_label"]
    if data_col_p_x_leq_x_name is None:
        data_col_p_x_leq_x_name = kwargs["y_label"]
    # Create data
    x = np.linspace(data_range[0], data_range[-1], data_samples)
    y = [data_func(x_i) for x_i in x]
    # Configure generic plot options
    kwargs.setdefault("draw_grid", True)
    kwargs.setdefault("limits_x", (-1, data_range[-1] + 1))
    _plot_generic_configuration(plt, **kwargs)
    # Plot data (add -1=0, max+1=1 points)
    plt.plot(
        [-1, *x, data_range[-1] + 1], [0, *y, 1], linewidth=line_width, color=line_color
    )
    # Show/export plot and return the data
    _plot_generic_show_export(plt, **kwargs)
    if data_return:
        return pd.DataFrame(
            data={data_col_x_name: x, data_col_p_x_leq_x_name: y}
        ).sort_values(by=[data_col_x_name], ascending=True)


def scatter_plot(
    data_list: list[tuple[dict[float, float], str, str]],
    **kwargs: Unpack[GenericPlotParams],
):
    """
    Draw scattered points of one or multiple data sets which is stored in data_list.
    """
    # Configure generic plot options
    kwargs.setdefault("title", "Scatter Plot")
    kwargs.setdefault("x_label", "$x$")
    kwargs.setdefault("y_label", "")
    kwargs.setdefault("draw_grid", True)
    kwargs.setdefault("draw_legend", "upper right")
    _plot_generic_configuration(plt, **kwargs)
    # Plot data
    for data, color, label in data_list:
        plt.scatter(list(data.keys()), list(data.values()), color=color, label=label)
    # Configure generic plot show/export options
    # type: ignore[misc]
    _plot_generic_show_export(plt, **kwargs)
