import os
import typing
from dataclasses import dataclass

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from rignak.logging_utils import logger
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap
from scipy import stats

CDOP_VRAD_CPT = os.path.join(os.path.dirname(__file__), 'cdop_vrad.cpt')

from functools import wraps
from rignak.lazy_property import LazyProperty


def plot_decorator(format_ax: bool = True):
    def get_wrapper(function: typing.Callable):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            kwargs = Kwargs(**kwargs)
            ax = self.ax if kwargs.ax is None else kwargs.ax
            results = function(self, *args, ax=ax, kwargs=kwargs)
            if format_ax:
                self.format_ax(ax, kwargs)
            self.show(export_filename=kwargs.export_filename, display=kwargs.display)
            return results

        return wrapper

    return get_wrapper


def read_cpt(file_path: str = CDOP_VRAD_CPT) -> LinearSegmentedColormap:
    cdict = {'red': [], 'green': [], 'blue': []}

    with open(file_path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('B') or line.startswith('F') or line.startswith('N'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                x1, r1, g1, b1, x2, r2, g2, b2 = map(float, parts[:8])
                cdict['red'].append((x1, r1 / 255.0, r1 / 255.0))
                cdict['green'].append((x1, g1 / 255.0, g1 / 255.0))
                cdict['blue'].append((x1, b1 / 255.0, b1 / 255.0))

    cmap = LinearSegmentedColormap('custom_cpt', cdict)
    return cmap


@dataclass
class Kwargs:
    vmin: typing.Optional[np.generic] = None
    vmax: typing.Optional[np.generic] = None
    xmin: typing.Optional[np.generic] = None
    xmax: typing.Optional[np.generic] = None
    ymin: typing.Optional[np.generic] = None
    ymax: typing.Optional[np.generic] = None
    cmap_name: typing.Optional[str] = None
    under_threshold_color: typing.Optional[str] = None
    over_threshold_color: typing.Optional[str] = None
    invalid_color: typing.Optional[str] = None
    export_filename: typing.Optional[str] = None
    display: bool = False
    title: str = ''
    labels: typing.Optional[typing.Sequence[str]] = None
    alpha: float = 0.8
    xscale: typing.Optional[str] = None
    yscale: typing.Optional[str] = None
    xlabel: typing.Optional[str] = None
    ylabel: typing.Optional[str] = None
    x_are_dates: bool = False
    bins: typing.Optional[int] = 20
    density: bool = False
    histtype: str = "bar"
    marker: str = '+'
    interpolation: str = "nearest"
    fmt: typing.Optional[str] = None  # ".1%"
    xticks: typing.Optional[np.ndarray] = None
    xticks_rotation: typing.Optional[np.generic] = 0
    yticks: typing.Optional[np.ndarray] = None
    epsilon: float = 1e-3
    extent: typing.Optional[typing.Tuple[np.generic, np.generic, np.generic, np.generic]] = None

    projection: str = 'merc'
    resolution: str = 'i'
    water_color: str = 'lightgray'
    earth_color: str = 'darkgray'
    latitude_stride: np.generic = 1
    longitude_stride: np.generic = 1
    color: typing.Optional[str] = None
    linestyle: str = 'solid'
    linewidth: str = 2
    orientation: str = 'horizontal'
    fraction: float = 0.046
    colorbar_scale: str = 'linear'
    colorbar_display: bool = True

    positions: typing.Optional[np.ndarray] = None
    widths: typing.Optional[float]= None

    ax: typing.Optional[plt.Axes] = None
    axes: typing.Optional[typing.Sequence[plt.Axes]] = None
    grid: bool = True
    axis_display: bool = True

    _cmap: typing.Optional[LinearSegmentedColormap] = None
    _grid_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None

    aspect: typing.Optional[str] = None

    @property
    def colorbar(self) -> typing.Dict[str, typing.Any]:
        if self.colorbar_scale == 'log':
            norm = {'norm': matplotlib.colors.LogNorm(vmin=self.vmin, vmax=self.vmax)}
        else:
            norm = {'vmin': self.vmin, 'vmax': self.vmax}
        return norm

    @LazyProperty
    def grid_kwargs(self) -> typing.Dict[str, typing.Any]:
        return {}

    @LazyProperty
    def cmap(self) -> LinearSegmentedColormap:
        if self.cmap_name is not None and self.cmap_name.endswith('.cpt'):
            cmap = read_cpt(self.cmap_name)
        else:
            cmap = plt.get_cmap(self.cmap_name)
        cmap.set_extremes(under=self.under_threshold_color, over=self.over_threshold_color, bad=self.invalid_color)
        return cmap

    def cliping_transform(self, x: typing.Union[np.generic, np.ndarray]) -> typing.Union[np.generic, np.ndarray]:
        return np.clip(x, self.epsilon, 1 - self.epsilon)

    def logit_transform(self, x: typing.Union[np.generic, np.ndarray]) -> typing.Union[np.generic, np.ndarray]:
        def transform_without_scale(x: typing.Union[np.generic, np.ndarray]) -> typing.Union[np.generic, np.ndarray]:
            return np.log10(x / (1 - x))

        x = self.cliping_transform(x)
        x = transform_without_scale(x) - transform_without_scale(self.epsilon)
        return x


@dataclass
class Display:
    figsize: typing.Tuple[int, int] = (4, 4)
    fig: typing.Optional[plt.Figure] = None

    ncols: int = 1
    nrows: int = 1

    kwargs_keys: typing.Sequence[str] = ()

    _axes: typing.Optional[np.ndarray] = None
    ax: typing.Optional[matplotlib.axes.Axes] = None
    suptitle: str = ''

    def __getitem__(self, index: int) -> "Display":
        self.ax = self.axes[index]
        return self

    @LazyProperty
    def axes(self) -> np.ndarray:
        self.refresh_axes()
        return self._axes

    def refresh_axes(self) -> np.ndarray:
        self.fig, axs = plt.subplots(
            self.nrows,
            self.ncols,
            figsize=(self.figsize[0] * self.ncols, self.figsize[1] * self.nrows),
            layout='constrained'
        )
        if self.nrows == self.ncols == 1:
            axs = np.array([axs])
        self._axes = axs.flatten()
        self.ax = self.axes[0]
        return self._axes

    def show(self, display: bool = True, export_filename: typing.Optional[str] = None, close: bool = True) -> None:
        self.fig.suptitle(self.suptitle)

        if export_filename is not None:
            os.makedirs(os.path.dirname(export_filename), exist_ok=True)
            self.fig.savefig(export_filename)
        elif display:
            plt.show()
        if close:
            plt.close()

    def colorbar(self, img, ax, **kwargs) -> None:
        plt.colorbar(img, format=lambda x, _: f"{x:.2g}", **kwargs)

    @staticmethod
    def format_ax(ax: plt.Axes, kwargs: Kwargs) -> None:
        def set_scale() -> None:
            if kwargs.xscale is not None:
                ax.set_xscale(kwargs.xscale)
            if kwargs.yscale is not None:
                ax.set_yscale(kwargs.yscale)

        def set_labels() -> None:
            if kwargs.xlabel is not None:
                ax.set_xlabel(kwargs.xlabel)
            if kwargs.ylabel is not None:
                ax.set_ylabel(kwargs.ylabel)

        def set_ticks() -> None:
            if kwargs.xticks is not None:
                ax.set_xticks(kwargs.xticks)
            if kwargs.yticks is not None:
                ax.set_yticks(kwargs.yticks)

        def set_rotation() -> None:
            if kwargs.x_are_dates:
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                for label in ax.get_xticklabels():
                    kwargs.xticks_rotation = 45

            if kwargs.xticks_rotation:
                ax.tick_params(axis='x', rotation=kwargs.xticks_rotation)
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment('right')

        def set_grid() -> None:
            if kwargs.grid:
                ax.grid(visible=kwargs.grid, **kwargs.grid_kwargs)
            if kwargs.xscale == 'log':
                ax.grid(visible=kwargs.grid, which='minor', linestyle=':', axis='x')
            if kwargs.yscale == 'log':
                ax.grid(visible=kwargs.grid, which='minor', linestyle=':', axis='y')

        def set_limits() -> None:
            ax.set_xlim(kwargs.xmin, kwargs.xmax)
            ax.set_ylim(kwargs.ymin, kwargs.ymax)

        set_scale()
        set_labels()
        set_ticks()
        set_rotation()
        set_grid()
        set_limits()
        ax.set_title(kwargs.title)
        if kwargs.axis_display is False:
            ax.set_axis_off()

    @plot_decorator(format_ax=True)
    def plot(self, x: typing.Optional[np.ndarray], y: np.ndarray, ax: plt.Axes = None, kwargs: Kwargs = None) -> None:
        if x is None:
            x = range(len(y))

        ax.plot(
            x, y,
            label=kwargs.labels,
            color=kwargs.color,
            linestyle=kwargs.linestyle,
            linewidth=kwargs.linewidth,
            alpha=kwargs.alpha
        )
        if kwargs.labels:
            ax.legend()

    @plot_decorator(format_ax=True)
    def boxplot(self, x: typing.Optional[np.ndarray], y: np.ndarray, ax: plt.Axes = None,
                kwargs: Kwargs = None)  -> typing.Dict:
        boxplot = ax.boxplot(y, positions=kwargs.positions, widths=kwargs.widths)
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(boxplot[element], color=kwargs.color)

        if x is not None:
            ax.set_xticklabels(x)
        return boxplot

    def plot_several_y(self, x: typing.Optional[np.ndarray], ys: np.ndarray, **kwargs) -> None:
        ax = kwargs['ax'] if 'ax' in kwargs else None
        self.ax = ax

        for y in ys.T[::int(ys.shape[1] / 6)]:
            self.plot(x, y, **kwargs, linestyle=':', linewidth=1)

        self.plot(x, np.nanpercentile(ys, 25, axis=1), linestyle='--', color='black', labels='25%')
        self.plot(x, np.nanpercentile(ys, 50, axis=1), linestyle='-', color='black', labels='50%')
        self.plot(x, np.nanpercentile(ys, 75, axis=1), linestyle='--', color='black', labels='75%')

    @plot_decorator(format_ax=True)
    def scatter(self, x: np.ndarray, y: np.ndarray, ax: plt.Axes = None, kwargs: Kwargs = None) -> None:
        scatter_kwargs = dict(marker=kwargs.marker, color=kwargs.color)
        if isinstance(y, np.ndarray) and len(y.shape) == 2:
            for e in y:
                ax.scatter(x, e, **scatter_kwargs)
        else:
            ax.scatter(x, y, **scatter_kwargs)
        if kwargs.labels:
            ax.legend(kwargs.labels)

    @plot_decorator(format_ax=True)
    def plot_regression(self, x: np.ndarray, y: np.ndarray, ax: plt.Axes = None, kwargs: Kwargs = None) -> None:
        valid = np.logical_and(np.isfinite(x), np.isfinite(y))
        x = x[valid]
        y = y[valid]

        xy = np.vstack([x, y])
        kde = stats.gaussian_kde(xy)
        if kwargs.vmin is not None:
            kwargs.xmin = kwargs.vmin
            kwargs.ymin = kwargs.vmin
        else:
            if kwargs.xmin is None:
                kwargs.xmin = np.nanmin(x)
            if kwargs.ymin is None:
                kwargs.ymin = np.nanmin(y)

        if kwargs.vmax is not None:
            kwargs.xmax = kwargs.vmax
            kwargs.ymax = kwargs.vmax
        else:
            if kwargs.xmax is None:
                kwargs.xmax = np.nanmax(x)
            if kwargs.ymax is None:
                kwargs.ymax = np.nanmax(y)

        x_grid, y_grid = np.meshgrid(
            np.linspace(kwargs.xmin, kwargs.xmax, 100),
            np.linspace(kwargs.ymin, kwargs.ymax, 100)
        )
        z = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)
        levels = np.linspace(z.min(), z.max(), 100)

        ax.contourf(x_grid, y_grid, z, levels=100, cmap=kwargs.cmap, alpha=0.7)
        try:
            ax.contourf(x_grid, y_grid, z, levels=[z.min(), levels[1]], colors='white')
        except Exception as e:
            print(e, [z.min(), levels[1]])

        rmse = np.sqrt(np.mean((y - x) ** 2))
        bias = np.mean(y - x)
        pearson_corr = stats.pearsonr(x, y)

        textstr = (f'RMSE: {rmse:.2g}'
                   f'\nBias: {bias:.2g}'
                   f'\nPearson Corr: {pearson_corr.statistic:.1%}'
                   )
        ax.text(
            0.05, 0.95, textstr,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5)
        )

    @plot_decorator(format_ax=True)
    def imshow(self, array: np.ndarray, ax: plt.Axes = None, kwargs: Kwargs = None) -> None:
        img = ax.imshow(array, cmap=kwargs.cmap, interpolation=kwargs.interpolation,
                        aspect=kwargs.aspect, extent=kwargs.extent, **kwargs.colorbar)
        if kwargs.colorbar_display:
            self.colorbar(img, ax=ax, orientation=kwargs.orientation, fraction=kwargs.fraction)

    @plot_decorator(format_ax=True)
    def heatmap(self, array: np.ndarray, ax: plt.Axes = None, kwargs: Kwargs = None) -> None:
        if ax is None:
            ax = plt.subplot()
        if kwargs.fmt is None:
            kwargs.fmt = '.1%'

        seaborn.heatmap(array[::-1], annot=True, ax=ax, fmt=kwargs.fmt, cmap=kwargs.cmap, **kwargs.colorbar)
        kwargs.grid = False

        if kwargs.labels is not None:
            x = np.arange(len(kwargs.labels)) + .5
            ax.xaxis.set_ticks(x, kwargs.labels, rotation=30, horizontalalignment='right')
            ax.yaxis.set_ticks(x, kwargs.labels[::-1], rotation=0)
            ax.set_ylim(array.shape[0], 0)
            ax.set_xlim(array.shape[0], 0)

    def undecorated_mesh(
            self,
            array: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            ax: typing.Union[plt.Axes, Basemap] = None,
            kwargs: Kwargs = None
    ) -> None:
        img = ax.pcolormesh(x, y, array, cmap=kwargs.cmap, **kwargs.colorbar)
        if kwargs.aspect is not None:
            ax.set_aspect(kwargs.aspect)

        self.colorbar(img, ax=ax, orientation=kwargs.orientation, fraction=kwargs.fraction)
        ax.set_title(kwargs.title)

    @plot_decorator(format_ax=True)
    def imshow_mesh(
            self,
            array: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            ax: plt.Axes = None,
            kwargs: Kwargs = None
    ) -> None:
        self.undecorated_mesh(array, x, y, ax=ax, kwargs=kwargs)

    @plot_decorator(format_ax=False)
    def imshow_mercator(
            self,
            array: np.ndarray,
            lat_grid: np.ndarray,
            lon_grid: np.ndarray,
            ax: plt.Axes = None,
            kwargs: Kwargs = None
    ) -> Basemap:
        min_lat = np.nanmin(lat_grid) - 0.1
        max_lat = np.nanmax(lat_grid) + 0.1
        min_lon = np.nanmin(lon_grid) - 0.1
        max_lon = np.nanmax(lon_grid) + 0.1

        m = Basemap(
            projection=kwargs.projection,
            llcrnrlat=min_lat,
            urcrnrlat=max_lat,
            llcrnrlon=min_lon,
            urcrnrlon=max_lon,
            resolution=kwargs.resolution,
            ax=ax
        )

        if kwargs.latitude_stride is not None:
            parallels = np.arange(np.floor(min_lat), np.ceil(max_lat), kwargs.latitude_stride)
            m.drawparallels(parallels, labels=[1, 0, 0, 0])
        if kwargs.longitude_stride is not None:
            meridians = np.arange(np.floor(min_lon), np.ceil(max_lon), kwargs.longitude_stride)
            m.drawmeridians(meridians, labels=[0, 0, 0, 1])

        x, y = m(lon_grid, lat_grid)

        self.undecorated_mesh(array, x, y, ax=m, kwargs=kwargs)
        m.fillcontinents(color=kwargs.earth_color, lake_color=kwargs.water_color)
        m.drawmapboundary(fill_color=kwargs.water_color)
        return m

    @plot_decorator(format_ax=True)
    def barh(self, y: np.ndarray, x: np.ndarray, ax: plt.Axes, kwargs: Kwargs = None) -> None:
        if kwargs.xscale == "logit":
            kwargs.xscale = "linear"
            # logger(f"`logit` xscale not supported by plt for barh. Will use a data transformation.")
            if kwargs.xmin is not None or kwargs.xmax is not None:
                logger(f"{kwargs.xmin=} and {kwargs.xmax=} should be None. They will be defined from {kwargs.epsilon=}")

            kwargs.xmax = kwargs.logit_transform(1 - kwargs.epsilon)
            kwargs.xmin = 0
            x = kwargs.logit_transform(x)
            if kwargs.xlabel is None:
                kwargs.xlabel = 'logit'
            else:
                kwargs.xlabel += '(logit)'

        ax.barh(y, x, color=kwargs.color, alpha=kwargs.alpha)

    @plot_decorator(format_ax=True)
    def plot_histogram(
            self,
            data: typing.Sequence,
            ax: plt.Axes = None,
            kwargs: Kwargs = None
    ) -> None:
        if not isinstance(data, dict):
            data = {'': data}

        legends = []
        for i, (key, values) in enumerate(data.items()):
            legends.append(key + f" ({len(values)} points)")

            ax.hist(
                values,
                range=(kwargs.xmin, kwargs.xmax),
                bins=kwargs.bins,
                alpha=kwargs.alpha,
                density=kwargs.density,
                histtype=kwargs.histtype
            )

        if kwargs.ylabel is None:
            kwargs.ylabel = 'Number of points' if kwargs.density else 'Distribution'

        ax.legend(legends)

    @plot_decorator(format_ax=True)
    def plot_single_histogram(
            self,
            data: typing.Sequence,
            ax: plt.Axes = None,
            kwargs: Kwargs = None
    ) -> None:
        kwargs.ymin = 0
        kwargs.density = True

        if kwargs.xmax is None:
            kwargs.xmax = np.nanpercentile(data, 99)
        if kwargs.xmin is None:
            kwargs.xmin = np.nanpercentile(data, 1)
        # if kwargs.ymax is None:
        #     kwargs.ymax = kwargs.bins / (kwargs.xmax - kwargs.xmin) / 2

        data = data[np.isfinite(data)]
        nanmean = np.nanmean(data)
        nanstd = np.std(data)

        ax.hist(
            data,
            range=(kwargs.xmin, kwargs.xmax),
            bins=kwargs.bins,
            alpha=kwargs.alpha,
            density=kwargs.density,
            histtype=kwargs.histtype
        )

        ax.axvline(
            nanmean,
            color='r',
            linestyle='dashed',
            linewidth=2,
            label=f'Mean: {nanmean:.3g}'
        )

        x = (nanmean - nanstd, nanmean + nanstd)
        if kwargs.ymax is None:
            y = (ax.get_ylim()[1] * .75, ax.get_ylim()[1] * .75)
        else:
            y = (kwargs.ymax * .75, kwargs.ymax * .75)

        ax.plot(x, y, color='g', linewidth=2, label=f'Std: {nanstd:.3g}')

        ax.legend()
        if kwargs.xlabel is not None:
            ax.set_xlabel(kwargs.xlabel)
        if kwargs.ylabel is None:
            kwargs.ylabel = 'Number of points' if kwargs.density else 'Distribution'

    @staticmethod
    def rescale_axis(self, ax: plt.Axes, x_label: str = '', y_label: str = '', factor: np.generic = 1) -> None:
        if x_label:
            xticks = ax.get_xticks()
            ax.set_xticklabels([f'{x * factor}' for x in xticks])
            ax.set_xlabel(x_label)
        if y_label:
            yticks = ax.get_yticks()
            ax.set_yticklabels([f'{y * factor}' for y in yticks])
            ax.set_ylabel(y_label)
