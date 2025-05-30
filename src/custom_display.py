import os
from dataclasses import dataclass, field # Added field for default_factory
from pathlib import Path
from typing import ( 
    Optional, Callable, Any, Union, Dict, Sequence, Tuple, TypeVar, List, Type
)
from functools import wraps 

import matplotlib.axes
import matplotlib.colors # For LogNorm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.collections import PathCollection 
from matplotlib.image import AxesImage 
from matplotlib.container import BarContainer 
from matplotlib.contour import QuadContourSet 
from matplotlib.cm import ScalarMappable

import numpy as np

import seaborn 
from .logging_utils import logger
from matplotlib.colors import LinearSegmentedColormap 
from mpl_toolkits.basemap import Basemap 
from scipy import stats
from scipy.stats._stats_py import PearsonRResult


from .lazy_property import LazyProperty 

CDOP_VRAD_CPT: str = os.path.join(os.path.dirname(__file__), 'cdop_vrad.cpt')

_KwargsType = TypeVar('_KwargsType', bound='Kwargs')
_DisplayType = TypeVar('_DisplayType', bound='Display')


def plot_decorator(format_ax: bool = True) -> Callable[
    [Callable[..., Any]], Callable[..., Any] 
]:
    def get_wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapper(instance: 'Display', *args: Any, **call_kwargs: Any) -> Any:
            # instance is 'self' of the Display class method
            
            # Extract 'ax' and 'kwargs' (Kwargs object) from call_kwargs if they were passed by name.
            # If not, they will be None here.
            ax_from_call = call_kwargs.pop('ax', None)
            kwargs_obj_from_call = call_kwargs.pop('kwargs', None)

            # Determine the Kwargs object to use
            if isinstance(kwargs_obj_from_call, Kwargs):
                current_kwargs_obj = kwargs_obj_from_call
                # Update this Kwargs object with any remaining call_kwargs (fields of Kwargs)
                for field_name, field_value in call_kwargs.items():
                    if hasattr(current_kwargs_obj, field_name):
                        setattr(current_kwargs_obj, field_name, field_value)
            else:
                # If Kwargs object wasn't passed, create one from all call_kwargs
                # This assumes call_kwargs contains fields for Kwargs dataclass
                current_kwargs_obj = Kwargs(**call_kwargs)

            # Determine the Axes object to use
            ax_to_use: Optional[matplotlib.axes.Axes] = ax_from_call if ax_from_call is not None else instance.ax
            if ax_to_use is None: # Still None, try to get the first one from Display's axes
                if instance._axes is None or len(instance._axes) == 0 : instance.refresh_axes()
                ax_to_use = instance.axes[0] if instance.axes is not None and len(instance.axes) > 0 else None
            
            if ax_to_use is None:
                raise ValueError("Axes object not available for plotting and could not be initialized.")

            # Call the actual plotting function (e.g., Display.plot)
            # The *args are the main data arguments (like x, y for plot method)
            results = function(instance, *args, ax=ax_to_use, kwargs=current_kwargs_obj)
            
            if format_ax:
                instance.format_ax(ax_to_use, current_kwargs_obj)
            
            return results
        return wrapper
    return get_wrapper


def read_cpt(file_path: str = CDOP_VRAD_CPT) -> LinearSegmentedColormap:
    cdict: Dict[str, List[Tuple[float, float, float]]] = {'red': [], 'green': [], 'blue': []}

    with open(file_path) as f:
        for line in f:
            if line.startswith(('#', 'B', 'F', 'N')):
                continue
            parts: List[str] = line.split()
            if len(parts) >= 8:
                vals = [float(p) for p in parts[:8]]
                x1, r1, g1, b1 = vals[0], vals[1]/255.0, vals[2]/255.0, vals[3]/255.0
                # x2, r2, g2, b2 = vals[4], vals[5]/255.0, vals[6]/255.0, vals[7]/255.0 # x2,c2 not used per original
                
                # Assuming x1 is already scaled (e.g. 0 to 1). If CPT uses 0-100, x1 might need x1/100.
                # Original LinearSegmentedColormap input format: (value, color_left_of_value, color_right_of_value)
                # Here, it's (value, color_at_value, color_at_value) creating steps.
                cdict['red'].append((x1, r1, r1))
                cdict['green'].append((x1, g1, g1))
                cdict['blue'].append((x1, b1, b1))
                
    for color_key in cdict:
        cdict[color_key].sort(key=lambda tup: tup[0])
        # Optional: Add 0.0 and 1.0 end points if missing, using nearest color.
        # This depends on CPT file conventions.
        if not cdict[color_key] or cdict[color_key][0][0] > 0.0:
            first_color = cdict[color_key][0][1:] if cdict[color_key] else (0.0,0.0) # Default to black if empty
            cdict[color_key].insert(0, (0.0, first_color[0], first_color[1]))
        if cdict[color_key][-1][0] < 1.0:
            last_color = cdict[color_key][-1][1:]
            cdict[color_key].append((1.0, last_color[0], last_color[1]))


    cmap = LinearSegmentedColormap('custom_cpt', cdict)
    return cmap


@dataclass
class Kwargs:
    vmin: Optional[Union[int, float, np.generic]] = None
    vmax: Optional[Union[int, float, np.generic]] = None
    xmin: Optional[Union[int, float, np.generic]] = None
    xmax: Optional[Union[int, float, np.generic]] = None
    ymin: Optional[Union[int, float, np.generic]] = None
    ymax: Optional[Union[int, float, np.generic]] = None
    
    cmap_name: Optional[str] = None # if None, cmap property will use 'viridis'
    under_threshold_color: Optional[str] = 'cyan'
    over_threshold_color: Optional[str] = 'magenta'
    invalid_color: Optional[str] = 'gray'
    
    export_filename: Optional[str] = None
    display: bool = True
    title: str = ''
    labels: Optional[Sequence[str]] = None # E.g. for legend
    alpha: float = 0.8
    
    xscale: Optional[str] = None # E.g., 'linear', 'log'
    yscale: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    x_are_dates: bool = False # For date formatting on x-axis
    
    bins: Optional[Union[int, Sequence[float]]] = 20 # Can be int or sequence of bin edges
    density: bool = False
    histtype: str = "bar" # E.g., 'bar', 'barstacked', 'step', 'stepfilled'
    marker: str = '+'
    interpolation: str = "nearest" # For imshow
    fmt: Optional[str] = None # Format string for annotations, e.g., in heatmap
    
    xticks: Optional[np.ndarray] = None
    xticks_rotation: Optional[Union[int, float]] = 0.0
    yticks: Optional[np.ndarray] = None
    epsilon: float = 1e-3 # For logit transform clipping
    
    extent: Optional[Tuple[float, float, float, float]] = None # Extent for imshow: left, right, bottom, top

    projection: str = 'merc'
    resolution: str = 'i' # 'c', 'l', 'i', 'h', 'f'
    water_color: str = 'lightblue' # Changed from lightgray for better visibility
    earth_color: str = 'lightgreen' # Consider 'tan' or 'beige' for earth # Changed from darkgray
    latitude_stride: Union[int, float] = 1.0 # np.generic to float
    longitude_stride: Union[int, float] = 1.0 # np.generic to float
    
    color: Optional[str] = None # For line plots, scatter plots etc.
    linestyle: str = 'solid'
    linewidth: Union[int, float] = 2.0 
    
    orientation: str = 'horizontal' # For colorbar # 'vertical' or 'horizontal'
    fraction: float = 0.046 # Fraction of original axes to use for colorbar
    colorbar_scale: str = 'linear' # 'linear' or 'log'
    colorbar_display: bool = True

    positions: Optional[np.ndarray] = None # Boxplot specific
    widths: Optional[Union[float, np.ndarray]] = None # Can be scalar or array # Boxplot specific

    ax: Optional[matplotlib.axes.Axes] = None # Allow passing a specific Axes object
    axes: Optional[Sequence[matplotlib.axes.Axes]] = None # For multi-plot setups
    grid: bool = True
    axis_display: bool = True # Whether to display axis lines/ticks

    _cmap: Optional[LinearSegmentedColormap] = field(init=False, repr=False, default=None) # Defaulting to None, will be set by cmap property.
    _grid_kwargs: Dict[str, Any] = field(default_factory=dict, repr=False) # default_factory needed for mutable types


    aspect: Optional[str] = 'auto' # 'auto', 'equal', or numeric

    @property
    def colorbar(self) -> Dict[str, Any]: # colorbar arguments for plt.colorbar
        norm_args: Dict[str, Any] = {}
        # Ensure vmin/vmax are not None before creating LogNorm if they are numbers
        valid_vmin = self.vmin if isinstance(self.vmin, (int, float, np.generic)) else None
        valid_vmax = self.vmax if isinstance(self.vmax, (int, float, np.generic)) else None

        if self.colorbar_scale == 'log':
            if valid_vmin is not None and valid_vmax is not None and float(valid_vmin) > 0 and float(valid_vmax) > 0: # LogNorm needs positive values
                norm_args['norm'] = matplotlib.colors.LogNorm(vmin=float(valid_vmin), vmax=float(valid_vmax))
            # else LogNorm cannot be created with non-positive or None limits, fallback to linear or default
        else: 
            if valid_vmin is not None: norm_args['vmin'] = valid_vmin
            if valid_vmax is not None: norm_args['vmax'] = valid_vmax
        return norm_args

    @LazyProperty
    def grid_kwargs(self) -> Dict[str, Any]:
        return dict(self._grid_kwargs) # Return a copy

    @LazyProperty
    def cmap(self) -> LinearSegmentedColormap:
        if self._cmap is not None: # Already materialized
            return self._cmap

        effective_cmap_name = self.cmap_name if self.cmap_name else 'viridis'
        
        try:
            if effective_cmap_name.endswith('.cpt'):
                self._cmap = read_cpt(effective_cmap_name)
            else:
                self._cmap = plt.get_cmap(effective_cmap_name)
        except Exception: # Fallback if cmap name is invalid or CPT file not found/parsable
            self._cmap = plt.get_cmap('viridis') 

        if self._cmap is not None: # Should always be true due to fallback
            current_cmap_copy = self._cmap._copy() # type: ignore # Work on a copy to avoid modifying global cmap registry
            current_cmap_copy.set_extremes(under=self.under_threshold_color, 
                                    over=self.over_threshold_color, 
                                    bad=self.invalid_color)
            self._cmap = current_cmap_copy
        return self._cmap # type: ignore # mypy might complain if it can't guarantee _cmap is set

    def cliping_transform(self, x: Union[np.generic, np.ndarray]) -> Union[np.generic, np.ndarray]:
        return np.clip(x, self.epsilon, 1.0 - self.epsilon)

    def logit_transform(self, x: Union[np.generic, np.ndarray]) -> Union[np.generic, np.ndarray]:
        def transform_without_scale(val: Union[np.generic, np.ndarray]) -> Union[np.generic, np.ndarray]:
            return np.log10(val / (1.0 - val)) # type: ignore

        x_clipped = self.cliping_transform(x)
        x_transformed = transform_without_scale(x_clipped)
        epsilon_transformed = transform_without_scale(np.array(self.epsilon)) # Ensure epsilon is array for subtraction if x is array
        
        return x_transformed - epsilon_transformed


@dataclass
class Display:
    figsize: Tuple[int, int] = (4, 4) # Width, Height in inches
    fig: Optional[Figure] = field(init=False, repr=False, default=None) 

    ncols: int = 1
    nrows: int = 1

    kwargs_keys: Sequence[str] = field(default_factory=list) # Default to empty list

    _axes: Optional[np.ndarray] = field(init=False, repr=False, default=None) 
    ax: Optional[matplotlib.axes.Axes] = field(init=False, repr=False, default=None) # Current active Axes
    suptitle: str = ''

    def __post_init__(self: _DisplayType) -> None:
        if self.fig is None: # Initialize on creation
            self.refresh_axes()

    def __getitem__(self: _DisplayType, index: Union[int, Tuple[int, int]]) -> _DisplayType:
        if self._axes is None: self.refresh_axes()
        
        current_axes_array = self.axes # Access via property to ensure it's initialized
        try:
            if isinstance(index, int): # Flattened index
                self.ax = current_axes_array.flat[index]
            elif isinstance(index, tuple) and len(index) == 2: # (row, col) index
                # Need to reshape _axes back to 2D to use (row, col) if it was flattened
                axes_2d = current_axes_array.reshape(self.nrows, self.ncols)
                self.ax = axes_2d[index[0], index[1]]
            else:
                raise TypeError(f"Invalid index type {type(index)}. Must be int or Tuple[int, int].")
        except IndexError:
            raise IndexError(f"Axes index {index} out of bounds for shape ({self.nrows}, {self.ncols}).")
        return self

    @LazyProperty
    def axes(self: _DisplayType) -> np.ndarray: 
        if self._axes is None: self.refresh_axes()
        return self._axes # type: ignore # _axes is guaranteed to be set by refresh_axes

    def refresh_axes(self: _DisplayType) -> np.ndarray: 
        if self.fig is not None: # Close existing figure to avoid memory leaks if recreating
            plt.close(self.fig)

        fig_width = self.figsize[0] * self.ncols
        fig_height = self.figsize[1] * self.nrows
        self.fig, axs = plt.subplots(
            self.nrows, self.ncols,
            figsize=(fig_width, fig_height),
            layout='constrained' # 'constrained' is good, 'tight' is another option via plt.tight_layout()
        )
        if not isinstance(axs, np.ndarray): # Single Axes case
            self._axes = np.array([axs])
        elif self.nrows == 1 or self.ncols == 1: # If 1D array of axes
            self._axes = np.array(axs).flatten() # Flatten to make it consistently 1D for _axes
        else: # 2D array of axes
            self_axes_list = []
            for row in axs: # type: ignore
                for item_ax in row: # type: ignore
                    self_axes_list.append(item_ax)
            self._axes = np.array(self_axes_list) # Store as flattened array
            
        self.ax = self._axes[0] if self._axes is not None and len(self._axes) > 0 else None
        return self._axes # type: ignore

    def show(self: _DisplayType, display: bool = True, export_filename: Optional[str] = None, close: bool = True) -> None:
        if self.fig is None: 
            return

        if self.suptitle: # Add suptitle only if it's set
            self.fig.suptitle(self.suptitle)

        if export_filename:
            export_path = Path(export_filename)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(export_path)
        elif display: # Only show if not exporting and display is True
            plt.show()
        
        if close: # Close the figure after showing or saving
            plt.close(self.fig)
            self.fig = None # Mark as closed

    def colorbar(
        self: _DisplayType, 
        img: ScalarMappable, # Type for 'img' that can be used with colorbar
        ax: matplotlib.axes.Axes, 
        **kwargs: Any 
    ) -> None:
        cbar_format = kwargs.pop('format', "%.2g") # Use pop to remove if present, else default
        if self.fig is not None:
            # Use plt.colorbar attached to the figure of the axes for better layout control
            self.fig.colorbar(img, ax=ax, format=cbar_format, **kwargs)
        # else: # Fallback if no fig, though less ideal
            # plt.colorbar(img, ax=ax, format=cbar_format, **kwargs)


    @staticmethod
    def format_ax(ax: matplotlib.axes.Axes, kwargs: Kwargs) -> None:
        # Helper to avoid repeating None checks
        def set_if_not_none(setter: Callable[[Any], None], value: Optional[Any]) -> None:
            if value is not None: setter(value)

        set_if_not_none(ax.set_xscale, kwargs.xscale)
        set_if_not_none(ax.set_yscale, kwargs.yscale)
        set_if_not_none(ax.set_xlabel, kwargs.xlabel)
        set_if_not_none(ax.set_ylabel, kwargs.ylabel)
        
        if kwargs.xticks is not None: ax.set_xticks(kwargs.xticks)
        if kwargs.yticks is not None: ax.set_yticks(kwargs.yticks)

        effective_rotation = kwargs.xticks_rotation if kwargs.xticks_rotation is not None else (45.0 if kwargs.x_are_dates else 0.0)
        if kwargs.x_are_dates: # For date formatting on x-axis
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d')) # type: ignore

        if effective_rotation != 0.0:
            ax.tick_params(axis='x', rotation=float(effective_rotation))
            if float(effective_rotation) > 0 : # Adjust alignment for positive rotation
                 for label in ax.get_xticklabels(): label.set_horizontalalignment('right')
        
        if kwargs.grid:
            # Use a copy of grid_kwargs to avoid modifying the Kwargs instance's default
            grid_p = dict(kwargs.grid_kwargs if kwargs.grid_kwargs else {})
            ax.grid(**grid_p) # type: ignore #Issue with ** unpacking TypedDict vs Dict
            if kwargs.xscale == 'log':
                ax.grid(visible=kwargs.grid, which='minor', linestyle=':', axis='x', **grid_p) # type: ignore
            if kwargs.yscale == 'log':
                ax.grid(visible=kwargs.grid, which='minor', linestyle=':', axis='y', **grid_p) # type: ignore

        ax.set_xlim(kwargs.xmin, kwargs.xmax) # Handles None internally
        ax.set_ylim(kwargs.ymin, kwargs.ymax) # Handles None internally
        
        set_if_not_none(ax.set_title, kwargs.title)
        if not kwargs.axis_display: # Check boolean flag correctly
            ax.set_axis_off()

    @plot_decorator(format_ax=True)
    def plot(
        self: _DisplayType, 
        x: Optional[np.ndarray], 
        y: np.ndarray, 
        ax: Optional[matplotlib.axes.Axes] = None, # Provided by decorator
        kwargs: Optional[Kwargs] = None # Provided by decorator
    ) -> None:
        # kwargs and ax are guaranteed by decorator if it's working as intended.
        if kwargs is None: kwargs = Kwargs() # Should not happen with decorator
        if ax is None: raise ValueError("Axes not provided by decorator for plot")

        effective_x = x if x is not None else np.arange(len(y))
        label = kwargs.labels[0] if kwargs.labels and len(kwargs.labels) > 0 else None # Use first label if sequence
        ax.plot(effective_x, y, label=label, color=kwargs.color, linestyle=kwargs.linestyle,
                linewidth=float(kwargs.linewidth), alpha=kwargs.alpha) # Ensure float
        if label: ax.legend() # Show legend if labels are provided

    @plot_decorator(format_ax=True)
    def boxplot(
        self: _DisplayType, 
        x_labels: Optional[Sequence[str]], # Labels for x-axis categories
        data_series: Union[np.ndarray, Sequence[np.ndarray]], # Data for boxes
        ax: Optional[matplotlib.axes.Axes] = None, # Provided by decorator
        kwargs: Optional[Kwargs] = None # Provided by decorator
    ) -> Optional[Dict[str, Any]]: 
        if kwargs is None: kwargs = Kwargs()
        if ax is None: raise ValueError("Axes not provided by decorator for boxplot")
        
        # ax.boxplot expects data where each column is a dataset, or a sequence of datasets.
        bp_dict: Dict[str, Any] = ax.boxplot(data_series, positions=kwargs.positions, widths=kwargs.widths)
        
        plot_color = kwargs.color if kwargs.color else 'blue' # Default color if not specified
        for element_key in bp_dict: # Iterate directly over keys in dict
            # Check if element exists in dict (e.g. means might not)
            if element_key in bp_dict: 
                plt.setp(bp_dict[element_key], color=plot_color)

        if x_labels:
            # If positions are used, xticks should align with positions.
            tick_pos = kwargs.positions if kwargs.positions is not None else np.arange(1, len(x_labels) + 1)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(x_labels)
        return bp_dict

    def plot_several_y(
        self: _DisplayType, 
        x: Optional[np.ndarray], 
        ys: np.ndarray, # Expects 2D array (series_length, num_series)
        **call_kwargs: Any # User-provided kwargs for plot settings
    ) -> None:
        # This method calls self.plot, which is decorated.
        # The decorator handles ax and Kwargs object.
        # We need to construct a Kwargs object from call_kwargs or use a default.
        
        # Base Kwargs from user call, or default if not provided
        base_plot_kwargs = Kwargs(**call_kwargs)
        # Determine ax to use for all plots in this method call
        ax_to_use_for_all = call_kwargs.get('ax', self.ax)
        if ax_to_use_for_all is None: # Still none, get default from display
            if self._axes is None or len(self._axes)==0: self.refresh_axes()
            ax_to_use_for_all = self.axes[0] if self.axes is not None and len(self.axes)>0 else None
        if ax_to_use_for_all is None: raise ValueError("Ax not available for plot_several_y")


        # Plot individual series (subset for clarity)
        num_series_to_plot = min(6, ys.shape[1]) # Plot up to 6 example series
        indices_to_plot = np.linspace(0, ys.shape[1] - 1, num_series_to_plot, dtype=int)

        for i in indices_to_plot:
            # For each call to self.plot, create/modify Kwargs for that specific line
            series_specific_kwargs_dict = base_plot_kwargs.__dict__.copy()
            series_specific_kwargs_dict.update({'linestyle': ':', 'linewidth': 1, 'alpha': 0.5, 'labels': None, 'title': ''})
            # Ensure color is part of the base or set a default for these thin lines
            if 'color' not in series_specific_kwargs_dict or series_specific_kwargs_dict['color'] is None:
                series_specific_kwargs_dict['color'] = 'gray' 
            self.plot(x, ys[:, i], ax=ax_to_use_for_all, kwargs=Kwargs(**series_specific_kwargs_dict))

        # Percentiles
        q25, q50, q75 = np.nanpercentile(ys, [25, 50, 75], axis=1) # Ensure percentile calculations handle potential all-NaN slices if any.
        
        self.plot(x, q25, ax=ax_to_use_for_all, kwargs=Kwargs(**{**base_plot_kwargs.__dict__, 'linestyle':'--', 'color':'black', 'labels':['25%'], 'title':''}))
        self.plot(x, q50, ax=ax_to_use_for_all, kwargs=Kwargs(**{**base_plot_kwargs.__dict__, 'linestyle':'-',  'color':'black', 'labels':['50% (Median)'], 'title':''}))
        # The last plot call's kwargs will be used by decorator for overall formatting/show
        self.plot(x, q75, ax=ax_to_use_for_all, kwargs=Kwargs(**{**base_plot_kwargs.__dict__, 'linestyle':'--', 'color':'black', 'labels':['75%'], 'title': base_plot_kwargs.title}))
        
        if ax_to_use_for_all.get_legend() is None and base_plot_kwargs.labels : # Add legend if not already added by last plot # Explains condition.
             ax_to_use_for_all.legend()


    @plot_decorator(format_ax=True)
    def scatter(
        self: _DisplayType, 
        x: np.ndarray, 
        y: np.ndarray, # Can be 1D or 2D (for multiple series)
        ax: Optional[matplotlib.axes.Axes] = None, 
        kwargs: Optional[Kwargs] = None
    ) -> Optional[PathCollection]: 
        if kwargs is None: kwargs = Kwargs()
        if ax is None: raise ValueError("Axes not provided by decorator for scatter")

        scatter_custom_kwargs = dict(marker=kwargs.marker, color=kwargs.color, alpha=kwargs.alpha)
        pc: Optional[PathCollection] = None # Store PathCollection if needed, though scatter usually just draws.
        if y.ndim == 2 and y.shape[0] == len(x): # if y has multiple columns for x # Multiple series in y
             for i in range(y.shape[1]): # Assuming y rows are series if 2D, or handle columns
                  pc = ax.scatter(x, y[:,i], **scatter_custom_kwargs) # type: ignore
        else: # Standard x, y scatter # Single series
             pc = ax.scatter(x, y, **scatter_custom_kwargs) # type: ignore
        
        if kwargs.labels: ax.legend(kwargs.labels if isinstance(kwargs.labels, list) else [kwargs.labels]) # If labels are provided (e.g. one label for all points, or per series if logic allows)
        return pc

    @plot_decorator(format_ax=True)
    def plot_regression(
        self: _DisplayType, 
        x: np.ndarray, 
        y: np.ndarray, 
        ax: Optional[matplotlib.axes.Axes] = None, 
        kwargs: Optional[Kwargs] = None
    ) -> Optional[QuadContourSet]: 
        if kwargs is None: kwargs = Kwargs()
        if ax is None: raise ValueError("Axes not provided by decorator for plot_regression")

        valid_mask: np.ndarray = np.logical_and(np.isfinite(x), np.isfinite(y))
        x_valid, y_valid = x[valid_mask], y[valid_mask]

        if len(x_valid) < 2 or len(y_valid) < 2: return None 
        xy = np.vstack([x_valid, y_valid])
        try: kde = stats.gaussian_kde(xy)
        except (np.linalg.LinAlgError, ValueError): return None # ValueError if not enough variation # Cannot proceed with KDE

        xmin_p, ymin_p = (kwargs.xmin if kwargs.xmin is not None else float(np.nanmin(x_valid))), (kwargs.ymin if kwargs.ymin is not None else float(np.nanmin(y_valid)))
        xmax_p, ymax_p = (kwargs.xmax if kwargs.xmax is not None else float(np.nanmax(x_valid))), (kwargs.ymax if kwargs.ymax is not None else float(np.nanmax(y_valid)))
        if kwargs.vmin is not None: xmin_p = ymin_p = float(kwargs.vmin)
        if kwargs.vmax is not None: xmax_p = ymax_p = float(kwargs.vmax)
        # Update Kwargs actual xmin, xmax, etc. for format_ax if they were derived
        kwargs.xmin, kwargs.ymin, kwargs.xmax, kwargs.ymax = xmin_p, ymin_p, xmax_p, ymax_p

        x_grid, y_grid = np.meshgrid(np.linspace(xmin_p, xmax_p, 100), np.linspace(ymin_p, ymax_p, 100))
        z_density = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)
        min_z, max_z = z_density.min(), z_density.max()
        levels = np.linspace(min_z, max_z, 100) if min_z < max_z else np.array([min_z, max_z]) # Ensure levels are sensible
        
        qcs: Optional[QuadContourSet] = None
        if len(levels) > 1 : # Ensure there's a range for contouring
            qcs = ax.contourf(x_grid, y_grid, z_density, levels=levels, cmap=kwargs.cmap, alpha=kwargs.alpha) # Use alpha
            # Highlight lowest density contour
            try: ax.contourf(x_grid, y_grid, z_density, levels=[min_z, levels[1] if len(levels)>1 else max_z], colors='white', alpha=1.0) # Ensure full alpha for this
            except ValueError: pass # Skip this highlight if it fails # If levels are too close or problematic

        rmse, bias = float(np.sqrt(np.mean((y_valid - x_valid) ** 2))), float(np.mean(y_valid - x_valid))
        pr_res: PearsonRResult = stats.pearsonr(x_valid, y_valid) if len(x_valid) >=2 else PearsonRResult(np.nan, np.nan) # type: ignore # scipy.stats.PearsonRResult or tuple # Useful context for pearson_result
        textstr = (f'RMSE: {rmse:.2g}\nBias: {bias:.2g}\nPearson Corr: {pr_res.statistic:.2%}') # Use .2% for two decimal places
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)) # Increased alpha for better readability
        return qcs


    @plot_decorator(format_ax=True)
    def imshow(
        self: _DisplayType, 
        array: np.ndarray, 
        ax: Optional[matplotlib.axes.Axes] = None, 
        kwargs: Optional[Kwargs] = None
    ) -> Optional[AxesImage]: 
        if kwargs is None: kwargs = Kwargs()
        if ax is None: raise ValueError("Axes not provided by decorator for imshow")

        img: AxesImage = ax.imshow(array, cmap=kwargs.cmap, interpolation=kwargs.interpolation,
                        aspect=kwargs.aspect, extent=kwargs.extent, **kwargs.colorbar) # Important detail: Pass vmin, vmax, norm from colorbar property
        if kwargs.colorbar_display: self.colorbar(img, ax=ax, orientation=kwargs.orientation, fraction=kwargs.fraction)
        return img

    @plot_decorator(format_ax=True)
    def heatmap(
        self: _DisplayType, 
        array: np.ndarray, 
        ax: Optional[matplotlib.axes.Axes] = None, 
        kwargs: Optional[Kwargs] = None
    ) -> None: # Explains return/side-effect: Returns Axes modified by seaborn
        if kwargs is None: kwargs = Kwargs()
        if ax is None: raise ValueError("Axes not provided by decorator for heatmap")
        
        fmt_val = kwargs.fmt if kwargs.fmt is not None else ('.2f' if np.issubdtype(array.dtype, np.floating) else 'd')
        # Seaborn heatmap uses vmin/vmax from its own kwargs, not from colorbar dict
        # It also has 'cbar' boolean.
        cbar_kws = {'orientation': kwargs.orientation, 'fraction': kwargs.fraction, 'format': fmt_val} if kwargs.orientation else None
        seaborn.heatmap(array[::-1], annot=True, ax=ax, fmt=fmt_val, cmap=kwargs.cmap, 
                        vmin=kwargs.vmin if isinstance(kwargs.vmin, (int,float)) else None, # Ensure vmin/vmax are numbers for heatmap
                        vmax=kwargs.vmax if isinstance(kwargs.vmax, (int,float)) else None,
                        cbar=kwargs.colorbar_display, cbar_kws=cbar_kws) # Important: Heatmap's own cbar args, not compatible with plt.colorbar dict
        kwargs.grid = False # Heatmap usually doesn't need a grid on top

        if kwargs.labels and len(kwargs.labels) > 0:
            num_labels = len(kwargs.labels)
            ticks = np.arange(num_labels) + 0.5 # Explains tick_positions
            if array.shape[1] == num_labels: ax.set_xticks(ticks, kwargs.labels, rotation=kwargs.xticks_rotation if kwargs.xticks_rotation is not None else 30.0, ha='right') # X-axis labels (columns)
            if array.shape[0] == num_labels: ax.set_yticks(ticks, kwargs.labels[::-1], rotation=0) # Y-axis labels (rows) - remember array is plotted [::-1]
            ax.set_ylim(array.shape[0], 0) # Match inverted array # Explains array.shape[0], 0
            ax.set_xlim(0, array.shape[1]) # Adjust limits to show full heatmap cells


    def undecorated_mesh(
            self: _DisplayType, array: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, # Renamed for clarity
            ax: Union[matplotlib.axes.Axes, Basemap], kwargs: Kwargs
    ) -> ScalarMappable: # Explains return type (ScalarMappable is broader): pcolormesh returns PolyCollection or QuadMesh (both ScalarMappable)
        # Important usage note: pcolormesh expects X, Y to define corners of quadrilaterals.
        img: ScalarMappable = ax.pcolormesh(x_coords, y_coords, array, cmap=kwargs.cmap, 
                                           shading='auto', **kwargs.colorbar) # Explains shading='auto': Often needed if X, Y are not edge coordinates # Important detail: Pass vmin, vmax, norm
        if kwargs.aspect and hasattr(ax, 'set_aspect'): ax.set_aspect(kwargs.aspect) # type: ignore # Explains hasattr: Basemap might not have set_aspect
        if kwargs.colorbar_display: # Explains condition: Only add colorbar if display is true
             cbar_ax = ax if isinstance(ax, matplotlib.axes.Axes) else getattr(ax, 'ax', None) # Important: Ensure ax for colorbar is matplotlib.axes.Axes, not Basemap directly if that causes issues # Explains getattr: Basemap.ax
             if cbar_ax: self.colorbar(img, ax=cbar_ax, orientation=kwargs.orientation, fraction=kwargs.fraction)
        if hasattr(ax, 'set_title'): ax.set_title(kwargs.title) # type: ignore # Explains hasattr: Basemap might handle title differently or not at all
        return img


    @plot_decorator(format_ax=True) # Explains decorator arg: format_ax=True for regular Axes
    def imshow_mesh(
            self: _DisplayType, array: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, # Renamed for clarity
            ax: Optional[matplotlib.axes.Axes] = None, kwargs: Optional[Kwargs] = None
    ) -> ScalarMappable: # Explains return type: Returns QuadMesh
        if kwargs is None: kwargs = Kwargs()
        if ax is None: raise ValueError("Axes not provided by decorator for imshow_mesh")
        return self.undecorated_mesh(array, x_coords, y_coords, ax=ax, kwargs=kwargs)

    @plot_decorator(format_ax=False) # Explains decorator arg: format_ax=False because Basemap handles its own formatting
    def imshow_mercator(
            self: _DisplayType, array: np.ndarray, lat_grid: np.ndarray, lon_grid: np.ndarray,
            ax: Optional[matplotlib.axes.Axes] = None, kwargs: Optional[Kwargs] = None # Important: Basemap will be created on this Axes
    ) -> Basemap:
        if kwargs is None: kwargs = Kwargs()
        if ax is None: raise ValueError("Axes not provided by decorator for imshow_mercator")

        min_lat,max_lat = float(np.nanmin(lat_grid))-0.1, float(np.nanmax(lat_grid))+0.1
        min_lon,max_lon = float(np.nanmin(lon_grid))-0.1, float(np.nanmax(lon_grid))+0.1

        m = Basemap(projection=kwargs.projection, llcrnrlat=min_lat, urcrnrlat=max_lat,
                    llcrnrlon=min_lon, urcrnrlon=max_lon, resolution=kwargs.resolution, ax=ax) # Explains ax=ax: Draw Basemap on the provided matplotlib Axes

        lat_s, lon_s = float(kwargs.latitude_stride), float(kwargs.longitude_stride)
        if lat_s > 0: m.drawparallels(np.arange(np.floor(min_lat), np.ceil(max_lat), lat_s), labels=[1,0,0,0]) # Explains labels: Left labels
        if lon_s > 0: m.drawmeridians(np.arange(np.floor(min_lon), np.ceil(max_lon), lon_s), labels=[0,0,0,1]) # Explains labels: Bottom labels
        
        x_map, y_map = m(lon_grid, lat_grid) 
        self.undecorated_mesh(array, x_map, y_map, ax=m, kwargs=kwargs) # Important detail: Pass Basemap instance as 'ax'
        m.fillcontinents(color=kwargs.earth_color, lake_color=kwargs.water_color)
        m.drawmapboundary(fill_color=kwargs.water_color)
        m.drawcountries(); m.drawcoastlines() # Explains choice: Optionally draw countries, coastlines
        # Useful: Title is handled by undecorated_mesh if ax=m supports set_title (Basemap does)
        return m


    @plot_decorator(format_ax=True)
    def barh(
        self: _DisplayType, y_categories: np.ndarray, x_values: np.ndarray, # Useful: Category names or positions for y-axis # Useful: Lengths of bars
        ax: Optional[matplotlib.axes.Axes] = None, kwargs: Optional[Kwargs] = None
    ) -> Optional[BarContainer]: 
        if kwargs is None: kwargs = Kwargs()
        if ax is None: raise ValueError("Axes not provided by decorator for barh")
        
        x_plot_values = x_values
        if kwargs.xscale == "logit":
            kwargs.xscale = "linear"  # Explains logic: Actual scale for barh will be linear
            if kwargs.xmin is not None or kwargs.xmax is not None: logger.warning("Logit barh xmin/xmax ignored.") # type: ignore
            kwargs.xmax = kwargs.logit_transform(np.array(1.0 - kwargs.epsilon)) # Explains np.array: Use array for transform
            kwargs.xmin = kwargs.logit_transform(np.array(kwargs.epsilon)) # Explains np.array: Use array for transform
            x_plot_values = np.array([kwargs.logit_transform(v) if np.isfinite(v) and 0 < v < 1 else np.nan for v in x_values]) # Important condition: Only transform finite positive values for logit
            kwargs.xlabel = f"{kwargs.xlabel or ''} (logit scale)".strip()

        bc: BarContainer = ax.barh(y_categories, x_plot_values, color=kwargs.color, alpha=kwargs.alpha) # Explains logic: Actual plot uses transformed values
        # Explains limitation: Set ticks to represent original scale if possible (complex for logit)
        # Explains current state: For now, ticks will be on the transformed (logit) scale.
        return bc

    @plot_decorator(format_ax=True)
    def plot_histogram( # Explains purpose: For comparing multiple distributions
            self: _DisplayType, data_dict: Dict[str, Sequence[Union[int, float]]], # Useful: Dict of datasets
            ax: Optional[matplotlib.axes.Axes] = None, kwargs: Optional[Kwargs] = None
    ) -> None:
        if kwargs is None: kwargs = Kwargs()
        if ax is None: raise ValueError("Axes not provided by decorator for plot_histogram")
        # Useful: `data` in original was Sequence, but usage implies Dict for multiple histograms
        # Changed to data_dict for clarity

        for key, values in data_dict.items():
            finite_vals = [v for v in values if np.isfinite(v)] # Important: Filter out NaNs before histogramming, as they cause issues or are ignored.
            if not finite_vals: continue # Explains continue: Skip if no valid data
            ax.hist(finite_vals, range=(kwargs.xmin, kwargs.xmax) if kwargs.xmin is not None and kwargs.xmax is not None else None, # Explains logic: Ensure range is tuple or None
                    bins=kwargs.bins if kwargs.bins is not None else 'auto', alpha=kwargs.alpha, # Explains choice: 'auto' is often a good default
                    density=kwargs.density, histtype=kwargs.histtype, label=key + f" ({len(finite_vals)} pts)") # Explains label: Add label for legend
        kwargs.ylabel = kwargs.ylabel if kwargs.ylabel else ('Density' if kwargs.density else 'Frequency') # Explains choice: More standard terms
        if data_dict: ax.legend() # Explains condition: Only show legend if there's something to label


    @plot_decorator(format_ax=True)
    def plot_single_histogram( # Explains purpose: For detailed view of one distribution
            self: _DisplayType, data: Sequence[Union[int, float]], # Useful: Single dataset
            ax: Optional[matplotlib.axes.Axes] = None, kwargs: Optional[Kwargs] = None
    ) -> None:
        if kwargs is None: kwargs = Kwargs()
        if ax is None: raise ValueError("Axes not provided by decorator for plot_single_histogram")
        kwargs.ymin = 0.0 # Explains: Histograms start at 0 count/density
        # kwargs.density = True # Explains choice: Often single histograms are density for shape analysis

        finite_data = np.array([v for v in data if np.isfinite(v)])
        if not finite_data.size: return 

        # Explains choice: Use percentiles to avoid extreme outliers skewing the view too much.
        eff_xmin = kwargs.xmin if kwargs.xmin is not None else float(np.nanpercentile(finite_data, 1))
        eff_xmax = kwargs.xmax if kwargs.xmax is not None else float(np.nanpercentile(finite_data, 99))
        # Important: Update Kwargs for format_ax to use these calculated limits if they were None
        if kwargs.xmin is None: kwargs.xmin = eff_xmin
        if kwargs.xmax is None: kwargs.xmax = eff_xmax
        
        nanmean, nanstd = float(np.nanmean(finite_data)), float(np.nanstd(finite_data))
        ax.hist(finite_data, range=(eff_xmin, eff_xmax), bins=kwargs.bins if kwargs.bins is not None else 'auto',
                alpha=kwargs.alpha, density=kwargs.density, histtype=kwargs.histtype)
        
        current_ymax = ax.get_ylim()[1]
        # Explains logic: Set ymax based on histogram output if not specified, to ensure stats lines are visible
        stat_line_y = (kwargs.ymax * 0.75) if kwargs.ymax is not None else (current_ymax * 0.75) # Explains choice: Heuristic: place stat lines at 75% of the (potentially new) max y if not set by user # Explains logic: If user set ymax, use that as reference

        ax.axvline(nanmean, color='r', linestyle='dashed', linewidth=kwargs.linewidth, label=f'Mean: {nanmean:.3g}') # Useful: Use from Kwargs
        ax.plot((nanmean - nanstd, nanmean + nanstd), (stat_line_y, stat_line_y), color='g', 
                linewidth=float(kwargs.linewidth), label=f'Std: {nanstd:.3g}')
        ax.legend()
        if kwargs.xlabel: ax.set_xlabel(kwargs.xlabel)
        kwargs.ylabel = kwargs.ylabel if kwargs.ylabel else ('Density' if kwargs.density else 'Frequency') # Explains logic: Set default y-label based on density
        # Explains interaction: format_ax will apply this ylabel.


    def rescale_axis( # Explains: Instance method
        self: _DisplayType, ax: matplotlib.axes.Axes, x_label: str = '', y_label: str = '', 
        factor: Union[int, float, np.generic] = 1.0
    ) -> None:
        current_factor = float(factor) # Explains float(factor): Ensure factor is float for division
        if current_factor == 0: return # Explains condition: Avoid division by zero

        if x_label:
            # Explains logic: Format labels to show rescaled values. Precision might need adjustment.
            ax.set_xticklabels([f'{x / current_factor:.2g}' for x in ax.get_xticks()]) # type: ignore # Explains type ignore: get_xticks returns list of floats
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_yticklabels([f'{y / current_factor:.2g}' for y in ax.get_yticks()]) # type: ignore # Explains type ignore: get_yticks returns list of floats
            ax.set_ylabel(y_label)
