import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class DynamicsPlotter:
    """
    A unified class for plotting complex dynamics structures such as orbits,
    Julia sets, and Fatou basins.
    
    This class maintains a consistent coordinate system (window) for the complex 
    plane, ensuring that boundaries align perfectly with generated arrays from 
    the dynamics module.
    
    Attributes:
        x_min (float): The minimum real coordinate.
        x_max (float): The maximum real coordinate.
        y_min (float): The minimum imaginary coordinate.
        y_max (float): The maximum imaginary coordinate.
        figsize (tuple): Tuple specifying the figure dimensions.
        use_latex (bool): Whether to use LaTeX for text rendering.
    """
    
    def __init__(self, x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, figsize=(6, 6), use_latex=False):
        """
        Initializes the plotting window for the complex plane.

        Args:
            x_min (float): Minimum real value.
            x_max (float): Maximum real value.
            y_min (float): Minimum imaginary value.
            y_max (float): Maximum imaginary value.
            figsize (tuple): Figure size for matplotlib.
            use_latex (bool): If True, uses LaTeX engine for text rendering (slower but professional).
                              If False, uses standard matplotlib text rendering (faster for previews).
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.figsize = figsize
        self.use_latex = use_latex
        self.extent = [self.x_min, self.x_max, self.y_min, self.y_max]

        self._setup_latex_style()

    def _setup_latex_style(self) -> None:
        """Configures matplotlib fonts and styling, optionally using LaTeX."""
        params = {
            "axes.labelsize": 12,           # Standard size for thesis
            "font.size": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": False,             # Grid off by default, enabled specifically for orbits
            "grid.alpha": 0.4,
            "grid.linestyle": "--",
            "grid.color": "gray"
        }
        
        if self.use_latex:
            params.update({
                "text.usetex": True,            # Use LaTeX to write all text
                "font.family": "serif",         # Use serif fonts
                "font.serif": ["Computer Modern Roman"],
            })
        else:
            params.update({
                "text.usetex": False,           # Use standard matplotlib text rendering
                "font.family": "sans-serif",    # Standard UI-friendly font
            })
            
        plt.rcParams.update(params)

    def _setup_figure(self, title=None, show_axis=True, show_axis_labels=True, show_grid=False):
        """Internal helper to setup the figure, axes labels, and title."""
        fig, ax = plt.subplots(figsize=self.figsize)
        if title:
            ax.set_title(title, fontsize=14)
            
        if show_axis:
            if show_axis_labels:
                ax.set_xlabel(r"$\mathrm{Re}(z)$")
                ax.set_ylabel(r"$\mathrm{Im}(z)$")
            if show_grid:
                ax.grid(True)
        else:
            ax.axis('off')
            
        return fig, ax

    def add_points(self, ax, points, label=None, marker='x', color='black', s=100, zorder=10, **kwargs):
        """
        Auxiliary method to overlay complex points on an existing axis.
        
        Args:
            ax (matplotlib.axes.Axes): The axis to plot on.
            points (list or np.ndarray): Complex numbers to plot.
            label (str, optional): Legend label for the points.
            marker (str): Matplotlib marker style.
            color (str): Matplotlib color.
            s (int): Marker size.
            zorder (int): Drawing order (higher is on top).
            **kwargs: Additional arguments passed to ax.scatter.
        """
        r_real = [np.real(p) for p in points]
        r_imag = [np.imag(p) for p in points]
        
        ax.scatter(r_real, r_imag, color=color, marker=marker, s=s, label=label, zorder=zorder, **kwargs)
        
        if label:
            ax.legend()
        
        return ax

    def plot_orbit(self, orbit_data, connect_lines=True, marker='o', color='black', title="Orbit", show_axis=True, show_axis_labels=False, show_grid=True):
        """
        Plots the forward orbit of a point on the complex plane.

        Args:
            orbit_data (np.ndarray): 1D array of complex numbers representing the orbit.
            connect_lines (bool): If True, draws lines between successive points.
            marker (str): Matplotlib marker style for the orbit points.
            color (str): Color of the markers and lines.
            title (str, optional): Title of the plot.
            show_axis (bool): If True, shows axes and tick marks.
            show_axis_labels (bool): If True, shows "Re(z)" and "Im(z)" labels.
            show_grid (bool): If True, displays a background grid.
            
        Returns:
            fig, ax: The matplotlib figure and axes objects.
        """
        fig, ax = self._setup_figure(title, show_axis, show_axis_labels, show_grid)
        
        real_parts = np.real(orbit_data)
        imag_parts = np.imag(orbit_data)
        
        if connect_lines:
            ax.plot(real_parts, imag_parts, color=color, alpha=0.5, linestyle='-', linewidth=1)
            
        ax.scatter(real_parts, imag_parts, color=color, marker=marker, zorder=5)
        
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        
        plt.tight_layout()
        return fig, ax

    def plot_escape_time_fractal(self, escape_data, cmap='binary', title="Escape Time Fractal", show_axis=True, show_axis_labels=False, show_colorbar=False, colorbar_label="Iterations to escape"):
        """
        Plots an escape time fractal (e.g., a Julia set) as a 2D image.

        Args:
            escape_data (np.ndarray): 2D integer array containing escape times.
            cmap (str): Matplotlib colormap string.
            title (str, optional): Title of the plot.
            show_axis (bool): If True, shows axes and tick marks.
            show_axis_labels (bool): If True, shows "Re(z)" and "Im(z)" labels.
            show_colorbar (bool): If True, displays a colorbar.
            colorbar_label (str): Label for the colorbar (if shown).

        Returns:
            fig, ax: The matplotlib figure and axes objects.
        """
        fig, ax = self._setup_figure(title, show_axis, show_axis_labels, show_grid=False)
        
        # origin='lower' ensures y_min is at the bottom, y_max at the top
        im = ax.imshow(escape_data, extent=self.extent, origin='lower', cmap=cmap)
        
        if show_colorbar:
            fig.colorbar(im, ax=ax, label=colorbar_label)
        
        plt.tight_layout()
        return fig, ax

    def plot_julia_set_from_escape_times(self, escape_data, max_iter, color='black', title="Julia Set (Escape Boundary)", show_axis=True, show_axis_labels=False):
        """
        Extracts and plots the Julia set from escape time data (typically for polynomials).
        
        For polynomials, the filled Julia set consists of points that do not escape 
        to infinity (i.e., those that reach max_iter). The Julia set is the boundary 
        of this filled set.

        Args:
            escape_data (np.ndarray): 2D integer array containing escape times.
            max_iter (int): The maximum iterations used when generating the escape data.
            color (str): Matplotlib color string for the Julia set.
            title (str, optional): Title of the plot.
            show_axis (bool): If True, shows axes and tick marks.
            show_axis_labels (bool): If True, shows "Re(z)" and "Im(z)" labels.

        Returns:
            fig, ax: The matplotlib figure and axes objects.
        """
        fig, ax = self._setup_figure(title, show_axis, show_axis_labels, show_grid=False)
        
        # Points that never escaped are considered "inside" the filled Julia set
        is_inside = (escape_data == max_iter)
        
        # Find boundaries by comparing each pixel to its neighbors.
        up = np.roll(is_inside, shift=-1, axis=0)
        down = np.roll(is_inside, shift=1, axis=0)
        left = np.roll(is_inside, shift=-1, axis=1)
        right = np.roll(is_inside, shift=1, axis=1)
        
        # A point is on the boundary if its inside/outside status differs from any neighbor
        boundary_mask = (is_inside != up) | (is_inside != down) | \
                        (is_inside != left) | (is_inside != right)
        
        # Remove edge artifacts
        boundary_mask[0, :] = False
        boundary_mask[-1, :] = False
        boundary_mask[:, 0] = False
        boundary_mask[:, -1] = False
        
        # Create an RGBA image: transparent everywhere, chosen color on boundaries
        import matplotlib.colors as mcolors
        rgba_color = mcolors.to_rgba(color)
        
        img = np.zeros((*escape_data.shape, 4))
        img[boundary_mask] = rgba_color
        
        ax.imshow(img, extent=self.extent, origin='lower')
        
        plt.tight_layout()
        return fig, ax

    def plot_fatou_basins(self, basin_data, cmap='Set1', title="Fatou Basins of Attraction", show_axis=True, show_axis_labels=True):
        """
        Plots the Fatou basins of attraction. 

        Args:
            basin_data (np.ndarray): 2D integer array containing basin indices.
            cmap (str or Colormap): Matplotlib colormap string. Defaults to 'Set1'.
            title (str, optional): Title of the plot.
            show_axis (bool): If True, shows axes and tick marks.
            show_axis_labels (bool): If True, shows "Re(z)" and "Im(z)" labels.

        Returns:
            fig, ax: The matplotlib figure and axes objects.
        """
        fig, ax = self._setup_figure(title, show_axis, show_axis_labels, show_grid=False)
        
        min_val = int(np.min(basin_data))
        max_val = int(np.max(basin_data))
        
        # Custom logic to ensure qualitative colormaps map sequentially 
        # and don't "stretch" across the min and max values.
        if isinstance(cmap, str):
            try:
                base_cmap = plt.get_cmap(cmap)
                if hasattr(base_cmap, 'colors'):
                    color_list = []
                    # -1 usually means "did not converge". We give it a dark grey background.
                    if min_val < 0:
                        color_list.append((1.0, 1.0, 1.0, 1.0)) 
                        start_idx = 0
                    else:
                        start_idx = min_val
                        
                    # Grab the first N colors sequentially from the colormap
                    for i in range(start_idx, max_val + 1):
                        color_list.append(base_cmap.colors[i % len(base_cmap.colors)])
                        
                    cmap = mcolors.ListedColormap(color_list)
            except Exception:
                pass # Fallback to default behavior if anything fails
        
        # BoundaryNorm guarantees that each integer gets exactly one distinct color bin
        bounds = np.arange(min_val, max_val + 2) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N if hasattr(cmap, 'N') else len(bounds)-1)
        
        # We use a discrete mapping approach via imshow
        ax.imshow(basin_data, extent=self.extent, origin='lower', cmap=cmap, norm=norm)
            
        plt.tight_layout()
        return fig, ax

    def plot_julia_set_from_basins(self, basin_data, color='black', title="Julia Set (Basin Boundaries)", show_axis=True, show_axis_labels=False):
        """
        Extracts and plots the Julia set as the boundary of the Fatou basins.
        
        In complex dynamics, the Julia set is precisely the common boundary 
        of the Fatou basins. This method finds the boundaries numerically 
        by detecting adjacent pixels belonging to different basins.

        Args:
            basin_data (np.ndarray): 2D integer array containing basin indices.
            color (str): Matplotlib color string for the Julia set.
            title (str, optional): Title of the plot.
            show_axis (bool): If True, shows axes and tick marks.
            show_axis_labels (bool): If True, shows "Re(z)" and "Im(z)" labels.

        Returns:
            fig, ax: The matplotlib figure and axes objects.
        """
        fig, ax = self._setup_figure(title, show_axis, show_axis_labels, show_grid=False)
        
        # Find boundaries by comparing each pixel to its neighbors.
        # np.roll shifts the array to easily compare adjacent pixels.
        up = np.roll(basin_data, shift=-1, axis=0)
        down = np.roll(basin_data, shift=1, axis=0)
        left = np.roll(basin_data, shift=-1, axis=1)
        right = np.roll(basin_data, shift=1, axis=1)
        
        # A point is on the boundary if its basin index differs from any neighbor
        boundary_mask = (basin_data != up) | (basin_data != down) | \
                        (basin_data != left) | (basin_data != right)
        
        # Remove edge artifacts caused by the wrap-around behavior of np.roll
        boundary_mask[0, :] = False
        boundary_mask[-1, :] = False
        boundary_mask[:, 0] = False
        boundary_mask[:, -1] = False
        
        # Create an RGBA image: transparent everywhere, chosen color on boundaries
        import matplotlib.colors as mcolors
        rgba_color = mcolors.to_rgba(color)
        
        # Initialize empty transparent image (height, width, 4 channels for RGBA)
        img = np.zeros((*basin_data.shape, 4))
        # Apply color only to the boundary pixels
        img[boundary_mask] = rgba_color
        
        ax.imshow(img, extent=self.extent, origin='lower')
        
        plt.tight_layout()
        return fig, ax

    def save_figure(self, fig, filename, dpi=300, bbox_inches='tight', **kwargs):
        """
        Saves the matplotlib figure to a file.
        
        This is highly recommended when exporting figures for LaTeX. 
        'tight' bounding box removes unnecessary white space around the image.
        
        Args:
            fig (matplotlib.figure.Figure): The figure object to save.
            filename (str): Output filename/path (e.g., 'report/images/julia.pdf').
            dpi (int): Resolution of the image. 300 is standard for print quality.
            bbox_inches (str): Bounding box style.
            **kwargs: Additional arguments passed to fig.savefig.
        """
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        print(f"Saved figure to: {filename}")