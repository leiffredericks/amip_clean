# plot_3_observation_maps with optional stippling or hatching

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from pathlib import Path

# %% Plot any number of model maps in four columns
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from pathlib import Path
# from matplotlib.colors import SymLogNorm
from matplotlib.colors import SymLogNorm, FuncNorm


def plot_model_maps(
    maps, model_names, lat, lon, *,
    title="", colorbar_label="", cmap="RdBu_r", vlim=None,
    robust_percentile=98, number_of_levels=21, central_longitude=210,
    savepath=None, dpi=200, color_scale="linear", linthresh=None, gamma=0.7, plot_method="contourf", clim=None, 
    ):
    """Plot maps (model, lat, lon) in four columns; return fig and flattened axes."""
    maps = np.asarray(maps, dtype=float)
    lat, lon = np.asarray(lat), np.asarray(lon)
    model_names = list(model_names)

    # Check that every panel has a name and uses the supplied spatial coordinates.
    if maps.ndim != 3:
        raise ValueError(f"maps has shape {maps.shape}; expected (model, lat, lon).")
    if maps.shape[0] != len(model_names) or len(model_names) == 0:
        raise ValueError("Provide at least one map and exactly one name per map.")
    if lat.ndim != 1 or lon.ndim != 1 or min(lat.size, lon.size) < 2:
        raise ValueError("lat and lon must be one-dimensional, with at least two points.")
    if maps.shape[1:] != (lat.size, lon.size):
        raise ValueError(f"Map shape {maps.shape[1:]} does not match lat/lon lengths.")

    # Use one symmetric color scale across all panels. The percentile only controls
    # color saturation; it is unrelated to statistical robustness or significance.
    if vlim is None:
        finite_values = np.abs(maps[np.isfinite(maps)])
        if finite_values.size == 0:
            raise ValueError("The maps contain no finite values.")
        vlim = np.percentile(finite_values, robust_percentile)
        if vlim == 0:
            vlim = np.max(finite_values)
        if vlim == 0:
            vlim = 1.0
    print(vlim)

    if not np.isfinite(vlim) or vlim <= 0:
        raise ValueError("vlim must be a positive, finite number.")

    # Default to symmetric limits; allow explicit bounds for one-sided quantities.
    vmin_plot, vmax_plot = (-vlim, vlim) if clim is None else clim

    # Select how values map to colors; all options preserve zero at the midpoint.
    norm = None

    if color_scale == "symlog":
        threshold = 0.01 * vlim if linthresh is None else linthresh
        norm = SymLogNorm(linthresh=threshold, vmin=-vlim, vmax=vlim, base=10)

    elif color_scale == "power":
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be positive and finite.")

        # Apply the power separately to magnitude and sign, keeping negative values.
        # gamma < 1 emphasizes small magnitudes; gamma = 1 gives linear scaling.
        def forward(x):
            return np.sign(x) * np.abs(x)**gamma

        def inverse(x):
            return np.sign(x) * np.abs(x)**(1 / gamma)

        norm = FuncNorm((forward, inverse), vmin=-vlim, vmax=vlim)

    elif color_scale != "linear":
        raise ValueError("color_scale must be 'linear', 'symlog', or 'power'.")

    # Space contour boundaries evenly in color space, then convert to physical units.
    levels = np.linspace(vmin_plot, vmax_plot, number_of_levels)
    if norm is not None:
        levels = norm.inverse(np.linspace(0, 1, number_of_levels))

    if plot_method not in ("contourf", "pcolormesh"):
        raise ValueError("plot_method must be 'contourf' or 'pcolormesh'.")

    # Add rows as needed, always retaining four columns. Reserve a fixed amount
    # of figure height for the title and colorbar; each row gets about 2.3 inches.
    n_models, n_columns = len(model_names), 4
    n_rows = (n_models + n_columns - 1) // n_columns
    figure_height = 2.3 * n_rows + 1.3
    projection = ccrs.Robinson(central_longitude=central_longitude)
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(17, figure_height),
        subplot_kw={"projection": projection}, squeeze=False,
    )
    axes = axes.ravel()


    # Draw each model with the same projection and contour levels. Hide unused
    # panels in the final row without changing the size of the remaining panels.
    for model_index, ax in enumerate(axes):
        if model_index >= n_models:
            ax.set_visible(False)
            continue

        # Close the longitude seam unless it is already included in the grid.
        # As in your original function, add_cyclic_point expects regular longitudes.
        model_map = np.ma.masked_invalid(maps[model_index])
        if np.isclose(abs(lon[-1] - lon[0]), 360.0):
            plot_map, plot_lon = model_map, lon
        else:
            plot_map, plot_lon = add_cyclic_point(model_map, coord=lon, axis=-1)

        # Contours use discrete levels; pcolormesh displays individual grid cells.
        # Both use the same color limits and optional symmetric-log normalization.
        if plot_method == "contourf":
            mappable = ax.contourf(
                plot_lon, lat, plot_map, levels=levels, cmap=cmap, norm=norm,
                extend="both", transform=ccrs.PlateCarree(),
            )
        else:
            # Use the original grid: pcolormesh does not need a duplicated cyclic column.
            # With cell-center coordinates, shading="auto" infers the cell boundaries.
            color_limits = {"vmin": vmin_plot, "vmax": vmax_plot} if norm is None else {}
            mappable = ax.pcolormesh(
                lon, lat, model_map, cmap=cmap, norm=norm, shading="auto",
                transform=ccrs.PlateCarree(), **color_limits,
            )
        ax.set_global()
        ax.coastlines(resolution="110m", linewidth=0.5, color="0.2")
        ax.gridlines(linewidth=0.3, color="0.35", alpha=0.4, linestyle=":")
        ax.set_title(f"{model_index + 1}. {model_names[model_index]}", fontsize=10, pad=4)

    # Convert fixed-inch margins into figure fractions so the title and colorbar
    # do not move excessively far from the maps when more rows are added.
    fig.suptitle(title, fontsize=16, y=1 - 0.15 / figure_height)
    fig.subplots_adjust(
        left=0.02, right=0.98, top=1 - 0.65 / figure_height,
        bottom=0.75 / figure_height, wspace=0.025, hspace=0.10,
    )
    colorbar_axis = fig.add_axes([0.22, 0.27 / figure_height, 0.56, 0.18 / figure_height])
    colorbar = fig.colorbar(mappable, cax=colorbar_axis, orientation="horizontal", extend="both")
    colorbar.set_label(colorbar_label, fontsize=11)
    colorbar.ax.tick_params(labelsize=9)

    # Saving is optional; the figure and axes remain available for notebook edits.
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {savepath}")

    return fig, axes

##


def plot_3_observation_maps(
    maps,
    lat,
    lon,
    panel_titles,
    *,
    figure_title="Observations",
    colorbar_labels=None,
    cmaps="RdBu_r",
    vlims=None,
    robust_percentile=98,
    number_of_levels=21,
    central_longitude=0,

    # Optional significance/robustness overlay
    robust_masks=None,
    overlay_style="stipple",          # "stipple" or "hatch"
    overlay_for="significant",        # "significant" or "not_significant"

    # Custom scatter stippling
    stipple_stride=3,
    stipple_size=2.0,
    stipple_color="black",
    stipple_alpha=0.65,
    stipple_marker=".",

    # Hatching
    hatch_pattern="....",
    hatch_color="black",
    hatch_linewidth=0.5,

    savepath=None,
    dpi=200,
    ):
    """
    Plot three observational maps vertically.

    Parameters
    ----------
    maps
        List or array containing three maps. Each map must have shape
        (lat, lon).

    lat, lon
        One-dimensional spatial coordinates.

    panel_titles
        Three panel titles in top-to-bottom order.

    colorbar_labels
        Three colorbar labels. Defaults to blank labels.

    cmaps
        One colormap name or a list of three colormap names.

    vlims
        Three symmetric absolute color limits. Use None for automatic
        robust limits. A single number applies to all three panels.

    robust_masks
        Optional Boolean or 0/1 significance/robustness mask.

        Accepted shapes are:

            (lat, lon)
                One common mask applied to all three panels.

            (3, lat, lon)
                A separate mask for each panel.

        True or 1 means significant/robust. NaNs are not overlaid.

    overlay_style
        "stipple" uses a custom scatter overlay.
        "hatch" uses contour hatching.

    overlay_for
        "significant" overlays True portions of robust_masks.
        "not_significant" overlays False portions of robust_masks.

    stipple_stride
        Plot one stipple point every N grid cells in both directions.
        Use 1 to consider every grid cell.

    savepath
        Optional output filename.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes.
    """

    # ---------------------------------------------------------
    # Prepare and validate maps
    # ---------------------------------------------------------

    if len(maps) != 3:
        raise ValueError(
            f"Expected three maps, received {len(maps)}"
        )

    if len(panel_titles) != 3:
        raise ValueError(
            "panel_titles must contain exactly three titles"
        )

    maps = np.stack(
        [np.asarray(panel_map) for panel_map in maps],
        axis=0,
    )

    lat = np.asarray(lat)
    lon = np.asarray(lon)

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError(
            "lat and lon must be one-dimensional"
        )

    if maps.shape[1:] != (len(lat), len(lon)):
        raise ValueError(
            f"Map shape {maps.shape[1:]} does not match "
            f"lat/lon lengths {(len(lat), len(lon))}"
        )

    # ---------------------------------------------------------
    # Prepare panel-specific plotting options
    # ---------------------------------------------------------

    if colorbar_labels is None:
        colorbar_labels = ["", "", ""]

    if len(colorbar_labels) != 3:
        raise ValueError(
            "colorbar_labels must contain exactly three labels"
        )

    if isinstance(cmaps, str):
        cmaps = [cmaps] * 3
    else:
        cmaps = list(cmaps)

    if len(cmaps) != 3:
        raise ValueError(
            "cmaps must be one name or a list of three names"
        )

    if vlims is None:
        vlims = [None, None, None]

    elif np.isscalar(vlims):
        vlims = [float(vlims)] * 3

    else:
        vlims = list(vlims)

    if len(vlims) != 3:
        raise ValueError(
            "vlims must be one number or a list of three values"
        )

    # ---------------------------------------------------------
    # Validate robustness overlay options
    # ---------------------------------------------------------

    if overlay_style not in {"stipple", "hatch"}:
        raise ValueError(
            "overlay_style must be 'stipple' or 'hatch'"
        )

    if overlay_for not in {
        "significant",
        "not_significant",
    }:
        raise ValueError(
            "overlay_for must be 'significant' "
            "or 'not_significant'"
        )

    if not isinstance(stipple_stride, (int, np.integer)):
        raise TypeError(
            "stipple_stride must be an integer"
        )

    if stipple_stride < 1:
        raise ValueError(
            "stipple_stride must be at least 1"
        )

    common_robust_mask = False

    if robust_masks is not None:
        robust_masks = np.asarray(robust_masks)

        if robust_masks.ndim == 2:

            if robust_masks.shape != (len(lat), len(lon)):
                raise ValueError(
                    f"2-D robust_masks has shape "
                    f"{robust_masks.shape}; expected "
                    f"{(len(lat), len(lon))}"
                )

            common_robust_mask = True

        elif robust_masks.ndim == 3:

            expected_shape = (
                3,
                len(lat),
                len(lon),
            )

            if robust_masks.shape != expected_shape:
                raise ValueError(
                    f"3-D robust_masks has shape "
                    f"{robust_masks.shape}; expected "
                    f"{expected_shape}"
                )

        else:
            raise ValueError(
                "robust_masks must have shape "
                "(lat, lon) or (3, lat, lon)"
            )

    longitude_is_cyclic = np.isclose(
        abs(lon[-1] - lon[0]),
        360.0,
    )

    # ---------------------------------------------------------
    # Create figure
    # ---------------------------------------------------------

    projection = ccrs.Robinson(
        central_longitude=central_longitude
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 11.5),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )

    axes = np.asarray(axes).ravel()

    # ---------------------------------------------------------
    # Plot panels
    # ---------------------------------------------------------

    for panel_index, ax in enumerate(axes):

        panel_map = maps[panel_index]
        panel_vlim = vlims[panel_index]

        # -----------------------------------------------------
        # Determine the color scale
        # -----------------------------------------------------

        if panel_vlim is None:

            finite_values = np.abs(
                panel_map[np.isfinite(panel_map)]
            )

            if finite_values.size == 0:
                raise ValueError(
                    f"Panel {panel_index + 1} "
                    "has no finite values"
                )

            panel_vlim = np.percentile(
                finite_values,
                robust_percentile,
            )

            if panel_vlim == 0:
                panel_vlim = np.max(finite_values)

            if panel_vlim == 0:
                panel_vlim = 1.0

        levels = np.linspace(
            -panel_vlim,
            panel_vlim,
            number_of_levels,
        )

        # -----------------------------------------------------
        # Add a cyclic longitude point
        # -----------------------------------------------------

        if longitude_is_cyclic:
            plot_map = panel_map
            plot_lon = lon

        else:
            plot_map, plot_lon = add_cyclic_point(
                panel_map,
                coord=lon,
                axis=-1,
            )

        # -----------------------------------------------------
        # Plot the filled field
        # -----------------------------------------------------

        mappable = ax.contourf(
            plot_lon,
            lat,
            plot_map,
            levels=levels,
            cmap=cmaps[panel_index],
            extend="both",
            transform=ccrs.PlateCarree(),
        )

        # -----------------------------------------------------
        # Optional significance/robustness overlay
        # -----------------------------------------------------

        if robust_masks is not None:

            if common_robust_mask:
                panel_mask = robust_masks
            else:
                panel_mask = robust_masks[panel_index]

            # True or nonzero values mean significant/robust.
            # NaNs are always excluded.
            mask_is_valid = np.isfinite(panel_mask)

            mask_is_true = (
                mask_is_valid &
                (panel_mask != 0)
            )

            if overlay_for == "significant":
                display_mask = mask_is_true

            else:
                display_mask = (
                    mask_is_valid &
                    ~mask_is_true
                )

            # Do not overlay locations where the map is missing.
            display_mask &= np.isfinite(panel_map)

            if np.any(display_mask):

                # ---------------------------------------------
                # Custom scatter stippling
                # ---------------------------------------------

                if overlay_style == "stipple":

                    sampled_mask = display_mask[
                        ::stipple_stride,
                        ::stipple_stride,
                    ]

                    sampled_lat = lat[
                        ::stipple_stride
                    ]

                    sampled_lon = lon[
                        ::stipple_stride
                    ]

                    lat_indices, lon_indices = np.where(
                        sampled_mask
                    )

                    ax.scatter(
                        sampled_lon[lon_indices],
                        sampled_lat[lat_indices],
                        s=stipple_size,
                        c=stipple_color,
                        alpha=stipple_alpha,
                        marker=stipple_marker,
                        linewidths=0,
                        transform=ccrs.PlateCarree(),
                        zorder=4,
                    )

                # ---------------------------------------------
                # Contour hatching
                # ---------------------------------------------

                elif overlay_style == "hatch":

                    hatch_map = np.where(
                        display_mask,
                        1.0,
                        np.nan,
                    )

                    if longitude_is_cyclic:
                        plot_hatch_map = hatch_map
                        plot_hatch_lon = lon

                    else:
                        (
                            plot_hatch_map,
                            plot_hatch_lon,
                        ) = add_cyclic_point(
                            hatch_map,
                            coord=lon,
                            axis=-1,
                        )

                    with mpl.rc_context({
                        "hatch.color": hatch_color,
                        "hatch.linewidth": hatch_linewidth,
                    }):

                        hatch_artist = ax.contourf(
                            plot_hatch_lon,
                            lat,
                            plot_hatch_map,
                            levels=[0.5, 1.5],
                            colors="none",
                            hatches=[hatch_pattern],
                            transform=ccrs.PlateCarree(),
                            zorder=4,
                        )

                        # Preserve hatch appearance across
                        # Matplotlib versions.
                        if hasattr(
                            hatch_artist,
                            "collections",
                        ):
                            for collection in (
                                hatch_artist.collections
                            ):
                                collection.set_edgecolor(
                                    hatch_color
                                )
                                collection.set_facecolor(
                                    "none"
                                )

        # -----------------------------------------------------
        # Geographic formatting
        # -----------------------------------------------------

        ax.set_global()

        ax.coastlines(
            resolution="110m",
            linewidth=0.55,
            color="0.2",
            zorder=5,
        )

        ax.gridlines(
            linewidth=0.3,
            color="0.35",
            alpha=0.4,
            linestyle=":",
        )

        panel_letter = chr(
            ord("a") + panel_index
        )

        ax.set_title(
            f"({panel_letter}) "
            f"{panel_titles[panel_index]}",
            fontsize=14,
            pad=5,
        )

        # -----------------------------------------------------
        # Individual panel colorbar
        # -----------------------------------------------------

        colorbar = fig.colorbar(
            mappable,
            ax=ax,
            orientation="horizontal",
            pad=0.035,
            fraction=0.055,
            aspect=35,
            extend="both",
        )

        colorbar.set_label(
            colorbar_labels[panel_index],
            fontsize=12,
        )

        colorbar.ax.tick_params(
            labelsize=9,
        )

    # ---------------------------------------------------------
    # Figure title
    # ---------------------------------------------------------

    fig.suptitle(
        figure_title,
        fontsize=16,
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    if savepath is not None:

        savepath = Path(savepath)

        savepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            savepath,
            dpi=dpi,
            bbox_inches="tight",
        )

        print(f"Saved: {savepath}")

    return fig, axes
# %%
