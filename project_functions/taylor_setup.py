# Taylor plotting block
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
import warnings

# Sixteen FILLED shapes; no unfilled +/x markers. The last is a four-point star.
# A model keeps its AMIP-assigned shape in the matched historical and arrow plots.
TAYLOR_SHAPES = ["o", "s", "^", "v", "<", ">", "D", "d", "p", "h", "H", "8", "P", "X", "*", (4, 1, 0)]


def prepare_taylor_tables(member_scores, mean_member_scores, metric, bias_key="bias_over_std_x"):
    """Collect existing scores; never compute scores of averaged maps.

    Radius = contrast_ratio = spatial std(model map) / spatial std(obs map).
    Angle = arccos(pattern_correlation). Bias is model minus observations.
    Means come directly from mean_member_scores: arithmetic means of scores,
    NOT mean polar angles, Fisher-z means, or scores of ensemble-mean maps.
    """
    experiments = ["amip_hist", "historical_matched", "historical"]
    member_rows, mean_rows, mixed_support = [], [], []
    for experiment in experiments:
        for model, scores in member_scores[experiment][metric].items():
            # Retain undefined values in the tables. Position filtering happens
            # during plotting; missing bias will be gray rather than treated as zero.
            corr = np.asarray(scores["pattern_correlation"], dtype=float)
            ratio = np.asarray(scores["contrast_ratio"], dtype=float)
            bias = np.asarray(scores[bias_key], dtype=float)
            if corr.ndim != 1 or corr.shape != ratio.shape or corr.shape != bias.shape:
                raise ValueError(f"{experiment}/{model}: expected matching 1D member-score arrays.")
            for e in range(len(corr)):
                member_rows.append((experiment, metric, model, e, corr[e], ratio[e], bias[e]))
            avg = mean_member_scores[experiment][metric][model]
            mean_rows.append((experiment, metric, model, avg["pattern_correlation"],
                              avg["contrast_ratio"], avg[bias_key], len(corr)))

            # The saved averages omit NaNs independently for each score. Flag
            # cases where the mean coordinates/bias therefore use different members.
            valid = [np.isfinite(values) for values in (corr, ratio, bias)]
            if not (np.array_equal(valid[0], valid[1]) and np.array_equal(valid[0], valid[2])):
                mixed_support.append(f"{experiment}/{model}")

    member_table = pd.DataFrame(member_rows, columns=["experiment", "metric", "model", "member",
                                                     "correlation", "std_ratio", "bias"])
    mean_table = pd.DataFrame(mean_rows, columns=["experiment", "metric", "model",
                                                 "correlation", "std_ratio", "bias", "n_members"])
    if mixed_support:
        warnings.warn("Saved mean coordinates/bias use different finite-member subsets for: "
                      + ", ".join(mixed_support), stacklevel=2)
    return member_table, mean_table


def taylor_positions(table):
    """Return theta (radians), radius, and a valid-position mask in table order."""
    corr = table["correlation"].to_numpy(dtype=float)
    radius = table["std_ratio"].to_numpy(dtype=float)
    valid = np.isfinite(corr) & np.isfinite(radius) & (radius >= 0) & (np.abs(corr) <= 1 + 1e-10)
    return np.arccos(np.clip(corr, -1, 1)), radius, valid


def taylor_background(points, title, *, rmax=None, corr_min=None, zoom=False, rms_levels=None, figsize=(14, 8.5)):
    """Make polar Taylor axes, a model-key panel, and the visible-point mask.

    None rmax includes every finite point plus padding. Without angular zoom,
    show correlations 0..1, or -1..1 if ANY point is negatively correlated.
    zoom=True trims unused low-correlation angles without excluding points.
    Explicit rmax/corr_min override automatic limits and warn about hidden points.
    The radial origin is always zero: no radial offset or nonlinear radial scale.
    """
    theta, radius, valid = taylor_positions(points)
    if not valid.any():
        raise ValueError("No finite Taylor coordinates. Check correlations and spatial standard deviations.")
    rmax = 1.12 * max(1.0, radius[valid].max()) if rmax is None else float(rmax)
    if not np.isfinite(rmax) or rmax <= 1:
        raise ValueError("rmax must exceed 1 so the observational reference remains visible.")

    # Correlation sets the angular extent through arccos, not a linear angle scale.
    # Automatic limits always retain OBS and all valid points, including negatives.
    if corr_min is not None:
        if not np.isfinite(corr_min) or not -1 <= corr_min < 1:
            raise ValueError("corr_min must satisfy -1 <= corr_min < 1.")
        theta_max = np.arccos(corr_min)
    elif zoom:
        theta_max = min(np.pi, max(np.deg2rad(30), theta[valid].max() + np.deg2rad(5)))
    else:
        theta_max = np.pi if np.any(theta[valid] > np.pi / 2) else np.pi / 2
    visible = valid & (radius <= rmax) & (theta <= theta_max + 1e-12)
    hidden, undefined = int((valid & ~visible).sum()), int((~valid).sum())
    if hidden:
        warnings.warn(f"Chosen extent hides {hidden} of {valid.sum()} finite points; means still use all members.",
                      stacklevel=2)
    if undefined:
        warnings.warn(f"{undefined} points have undefined/invalid Taylor coordinates and cannot be drawn.",
                      stacklevel=2)

    # Reserve the right panel for model identities, keeping legends off the data.
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(1, 2, width_ratios=[2.8, 1.6], left=0.06, right=0.98,
                           bottom=0.20, top=0.87, wspace=0.20)
    ax = fig.add_subplot(grid[0, 0], projection="polar")
    key_ax = fig.add_subplot(grid[0, 1])
    key_ax.set_axis_off()
    fig.suptitle(title, y=0.975, fontsize=15)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetalim(0, theta_max)
    ax.set_ylim(0, rmax)
    ax.set_rorigin(0)
    ax.set_rlabel_position(0)

    # Angular labels are correlations; radii are normalized spatial std, NOT RMS.
    corr_ticks = np.array([1, .99, .95, .9, .8, .6, .4, .2, 0, -.2, -.4, -.6, -.8, -.9, -.95, -.99, -1])
    angles = np.arccos(corr_ticks)
    inside = angles <= theta_max + 1e-12
    ax.set_thetagrids(np.degrees(angles[inside]), labels=[f"{v:g}" for v in corr_ticks[inside]])
    radial_ticks = MaxNLocator(nbins=5).tick_values(0, rmax)
    radial_ticks = radial_ticks[(radial_ticks > 0) & (radial_ticks < rmax)]
    ax.set_rticks(np.unique(np.r_[radial_ticks, 1.0]))
    ax.tick_params(labelsize=9)
    ax.grid(color="0.82", linewidth=0.65)
    ax.set_xlabel(r"Normalized spatial standard deviation  $\sigma_y/\sigma_x$", labelpad=25)
    ax.text(0.5, 1.095, "Spatial pattern correlation", ha="center", transform=ax.transAxes, fontsize=11)

    # Centered normalized RMSE is Euclidean distance from OBS in Taylor space:
    # E'^2 / sigma_x^2 = 1 + radius^2 - 2*radius*cos(theta).
    tt, rr = np.meshgrid(np.linspace(0, theta_max, 300), np.linspace(0, rmax, 260))
    error = np.sqrt(np.maximum(0, 1 + rr**2 - 2 * rr * np.cos(tt)))
    levels = MaxNLocator(nbins=6).tick_values(0, error.max()) if rms_levels is None else np.asarray(rms_levels)
    levels = np.unique(levels[(levels > 0) & (levels < error.max())])
    if len(levels):
        contours = ax.contour(tt, rr, error, levels=levels, colors="#58835C",
                              linewidths=0.8, linestyles="--", zorder=1)
        ax.clabel(contours, fmt="%g", fontsize=8)
    arc = np.linspace(0, theta_max, 300)
    ax.plot(arc, np.ones_like(arc), color="0.35", linewidth=1.2, zorder=1)
    ax.scatter([0], [1], marker="*", s=185, color="black", zorder=9, clip_on=False)
    ax.annotate("OBS", (0, 1), xytext=(8, -18), textcoords="offset points", fontsize=10)
    fig.text(0.06, 0.045, "Dashed green: centered RMSE / obs spatial SD.  "
             "Mean-score points summarize coordinates; their contour value is not mean member centered RMSE.",
             fontsize=9, color="0.3")
    fig.text(0.06, 0.022, f"View: {hidden} finite points outside limits; {undefined} undefined positions.",
             fontsize=8, color="0.4")
    return fig, ax, key_ax, visible


def taylor_bias_colors(values, bias_vlim=None, cmap="RdBu_r"):
    """One zero-centered linear scale; NaN bias is gray, not zero."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    limit = float(np.abs(finite).max()) if finite.size else 1.0
    if limit == 0:
        limit = 1.0                         # A useful color scale when every bias is zero.
    limit = limit if bias_vlim is None else float(bias_vlim)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("bias_vlim must be positive and finite.")
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("0.65")
    return color_map, mcolors.Normalize(vmin=-limit, vmax=limit)


def add_taylor_colorbar(fig, color_map, norm, bias_label):
    """Place one signed-bias scale below the Taylor axes."""
    cax = fig.add_axes([0.11, 0.115, 0.43, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_map), cax=cax,
                      orientation="horizontal", extend="both")
    cb.set_label(bias_label, fontsize=10)
    return cb


def label_taylor_models(ax, means, label_offsets):
    """Place numbers near their points, avoiding earlier labels when possible.

    Search a few rings of candidate text positions in SCREEN space. Only text
    moves; a thin leader connects each label to its unchanged Taylor coordinate.
    Explicit label_offsets[model] = (dx, dy) in points overrides the search.
    """
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    pixels_per_point = ax.figure.dpi / 72
    theta, radius, _ = taylor_positions(means)
    centers = ax.transData.transform(np.column_stack([theta, radius]))
    marker_radii = (np.sqrt(means["marker_area"].to_numpy()) / 2 + 2) * pixels_per_point
    marker_boxes = [Bbox.from_bounds(x - r, y - r, 2*r, 2*r) for (x, y), r in zip(centers, marker_radii)]
    # Reserve existing tick/contour/OBS labels as well as model markers.
    occupied = [text.get_window_extent(renderer).expanded(1.1, 1.3)
                for text in list(ax.texts) + ax.get_xticklabels() + ax.get_yticklabels()]
    obs_x, obs_y = ax.transData.transform((0, 1))
    obs_radius = 10 * pixels_per_point
    marker_boxes.append(Bbox.from_bounds(obs_x - obs_radius, obs_y - obs_radius, 2*obs_radius, 2*obs_radius))
    for i, row in enumerate(means.itertuples()):
        xy = (theta[i], radius[i])
        text = ax.annotate(str(row.number), xy, xytext=(10, 10), textcoords="offset points",
                           ha="center", va="center", fontsize=8, zorder=7,
                           path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
        # Prefer nearby labels, extending the search only when a local cluster
        # is crowded. This is a simple placement aid, not an optimal layout solver.
        if row.model in label_offsets:
            candidates = [label_offsets[row.model]]
        else:
            candidates = [(distance*np.cos(angle), distance*np.sin(angle))
                          for distance in [12, 20, 30, 42, 56, 72]
                          for angle in np.deg2rad([45, 135, -45, -135, 0, 90, 180, -90])]
        best, fewest = candidates[0], np.inf
        for offset in candidates:
            text.set_position(offset)
            box = text.get_window_extent(renderer).expanded(1.2, 1.3)
            collisions = sum(box.overlaps(other) for other in occupied + marker_boxes)
            # Stay within the actual polar wedge, not its rectangular bounding
            # box, so labels do not spill below the correlation=1 baseline.
            corners = [(box.x0, box.y0), (box.x0, box.y1), (box.x1, box.y0), (box.x1, box.y1)]
            collisions += 100 * (not ax.patch.contains_points(corners).all())
            if collisions < fewest:
                best, fewest = offset, collisions
            if collisions == 0:
                break
        text.set_position(best)
        occupied.append(text.get_window_extent(renderer).expanded(1.2, 1.3))
        ax.annotate("", xy, xytext=best, textcoords="offset points", zorder=4,
                    arrowprops=dict(arrowstyle="-", color="0.45", linewidth=0.5,
                                    shrinkA=4, shrinkB=np.sqrt(row.marker_area)/2))


def plot_taylor_experiment(member_table, mean_table, experiment, model_markers, *,
                           title=None, bias_label=r"Mean bias / obs spatial SD  $(\mu_y-\mu_x)/\sigma_x$",
                           bias_vlim=None, cmap="RdBu_r", rmax=None, corr_min=None, zoom=False,
                           rms_levels=None, member_alpha=0.22, member_size=28, mean_size=100,
                           bias_size_ref=None, large_group_bias="size", label_offsets=None, figsize=(14, 8.5)):
    """Draw <=16 models with shapes/colors, or >16 with numbered circles.

    For >16 models, large_group_bias="size" gives marker AREA proportional to
    absolute mean bias, with a visible area floor; background members are gray.
    Choose "color" for fixed-size means and signed-bias colors on BOTH members
    and means. Each point uses its own bias, with one common color scale.
    Returns fig, ax, model_key; model_key retains means hidden by a chosen crop.
    """
    if large_group_bias not in {"size", "color"}:
        raise ValueError('large_group_bias must be "size" or "color".')
    members = member_table.loc[member_table["experiment"] == experiment].copy()
    means = mean_table.loc[mean_table["experiment"] == experiment].copy().reset_index(drop=True)
    if means.empty:
        raise ValueError(f"No model means for {experiment}.")
    models = list(means["model"])
    if len(models) != len(set(models)):
        raise ValueError("Expected one mean row per model.")
    small_group = len(models) <= 16
    if small_group and not all(model in model_markers for model in models):
        raise ValueError("Supply a marker for every model in this <=16-model group.")
    points = pd.concat([members, means], ignore_index=True)
    title = f"{experiment} — {means['metric'].iloc[0]}" if title is None else title
    fig, ax, key_ax, visible = taylor_background(points, title, rmax=rmax, corr_min=corr_min,
                                                zoom=zoom, rms_levels=rms_levels, figsize=figsize)
    members["in_view"], means["in_view"] = visible[:len(members)], visible[len(members):]
    color_map, norm = taylor_bias_colors(points["bias"], bias_vlim, cmap)
    label_offsets = {} if label_offsets is None else label_offsets

    # Background members are drawn before every model summary. Whenever color
    # encodes bias, EACH MEMBER uses its own bias, not that model's mean bias.
    if small_group:
        for model in models:
            group = members.loc[(members["model"] == model) & members["in_view"]]
            theta, radius, _ = taylor_positions(group)
            colors = color_map(norm(np.ma.masked_invalid(group["bias"].to_numpy())))
            ax.scatter(theta, radius, marker=model_markers[model], s=member_size,
                       c=colors, alpha=member_alpha, linewidths=0, zorder=2)
        for row in means.loc[means["in_view"]].itertuples():
            color = color_map(norm(row.bias)) if np.isfinite(row.bias) else "0.65"
            ax.scatter([np.arccos(np.clip(row.correlation, -1, 1))], [row.std_ratio],
                       marker=model_markers[row.model], s=mean_size, color=color,
                       edgecolors="0.15", linewidths=0.8, zorder=5, clip_on=False)
        handles = [Line2D([], [], marker=model_markers[model], linestyle="none", markersize=8,
                          markerfacecolor="0.65", markeredgecolor="0.15", label=model) for model in models]
        key_ax.legend(handles=handles, title="Model shapes", loc="upper left", frameon=False,
                      fontsize=9, labelspacing=0.7)
        key_ax.text(0, 0.03, "Faint: individual members\nSolid: mean member scores\nGray fill: undefined bias",
                    transform=key_ax.transAxes, fontsize=9, va="bottom")
        add_taylor_colorbar(fig, color_map, norm, bias_label)
    else:
        group = members.loc[members["in_view"]]
        theta, radius, _ = taylor_positions(group)
        use_bias_color = large_group_bias == "color"
        member_colors = color_map(norm(np.ma.masked_invalid(group["bias"].to_numpy()))) if use_bias_color else "0.5"
        ax.scatter(theta, radius, s=member_size, color=member_colors, alpha=member_alpha,
                   linewidths=0, zorder=2)

        # Color mode keeps mean marker areas equal: color alone represents signed
        # bias. The optional size mode represents magnitude only, retaining a
        # positive area floor so a zero-bias model is still visible.
        means["number"] = np.arange(1, len(means) + 1)
        if use_bias_color:
            means["marker_area"] = float(mean_size)
        else:
            finite_bias = means.loc[np.isfinite(means["bias"]), "bias"].to_numpy()
            reference = float(np.abs(finite_bias).max()) if finite_bias.size else 1.0
            reference = 1.0 if reference == 0 else reference
            reference = reference if bias_size_ref is None else float(bias_size_ref)
            if not np.isfinite(reference) or reference <= 0:
                raise ValueError("bias_size_ref must be positive and finite.")
            means["marker_area"] = 45 + 180 * means["bias"].abs().fillna(0) / reference
        for row in means.loc[means["in_view"]].itertuples():
            xy = (np.arccos(np.clip(row.correlation, -1, 1)), row.std_ratio)
            fill = (color_map(norm(row.bias)) if np.isfinite(row.bias) else "0.65") if use_bias_color else "0.30"
            ax.scatter(*xy, s=row.marker_area, color=fill, edgecolors="black", linewidths=0.7, zorder=5)
        label_taylor_models(ax, means.loc[means["in_view"]], label_offsets)

        # Number key includes the SIGNED mean bias, which marker size cannot show.
        columns = 1 if len(means) <= 24 else 2
        rows_per_column = int(np.ceil(len(means) / columns))
        for index, row in enumerate(means.itertuples()):
            column, line = divmod(index, rows_per_column)
            signed = f"{row.bias:+.2f}" if np.isfinite(row.bias) else "NaN"
            key_ax.text(column * 0.54, 0.93 - line * 0.72 / max(rows_per_column, 1),
                        f"{row.number:02d}  {row.model}  ({signed})", fontsize=8, va="top",
                        transform=key_ax.transAxes)
        key_ax.text(0, 1, "Model number / name / signed mean bias", fontsize=10, va="top",
                    transform=key_ax.transAxes)
        if use_bias_color:
            add_taylor_colorbar(fig, color_map, norm, bias_label)
            key_ax.text(0, 0.03, "Faint: individual members\nNumbered: mean member scores\nGray fill: undefined bias",
                        transform=key_ax.transAxes, fontsize=9, va="bottom")
        else:
            sizes = np.array([0, reference / 2, reference])
            handles = [Line2D([], [], linestyle="none", marker="o", color="0.30", markeredgecolor="black",
                              markersize=np.sqrt(45 + 180 * value / reference), label=f"{value:.2g}") for value in sizes]
            key_ax.legend(handles=handles, title="Absolute mean bias (size)", loc="lower left",
                          frameon=False, ncol=3, fontsize=9, title_fontsize=10)
            fig.text(0.11, 0.125, "Bias values: " + bias_label + "\nFaint gray: members; numbered circles: mean member scores.",
                     fontsize=10, va="center")
    return fig, ax, means


def plot_taylor_shift(mean_table, model_markers, *, title="Historical → AMIP",
                      bias_label=r"Mean bias / obs spatial SD  $(\mu_y-\mu_x)/\sigma_x$",
                      bias_vlim=None, cmap="RdBu_r", rmax=None, corr_min=None, zoom=False,
                      rms_levels=None, mean_size=110, historical_edge="#252525", amip_edge="#009E73",
                      figsize=(14, 8.5)):
    """Connect matched MODEL mean-score coordinates, historical tail to AMIP head.

    No member-to-member pairing is assumed. Match model names, not row indices.
    Arrows are straight in the displayed Taylor plane, not linear in polar angle.
    Their displacement is not the RMSE between the historical and AMIP maps.
    """
    amip = mean_table.loc[mean_table["experiment"] == "amip_hist"].copy()
    historical = mean_table.loc[mean_table["experiment"] == "historical_matched"].copy()
    if set(amip["model"]) != set(historical["model"]) or amip.empty:
        raise ValueError("The arrow plot requires the same nonempty model set in AMIP and historical_matched.")
    if len(amip) > 16:
        raise ValueError("The paired-shape plot supports at most 16 models.")
    if not all(model in model_markers for model in amip["model"]):
        raise ValueError("A shared marker is required for every paired model.")
    # validate prevents accidental many-to-many merging if a model is duplicated.
    paired = historical.merge(amip, on="model", suffixes=("_historical", "_amip"), validate="one_to_one")
    paired = paired.set_index("model").loc[list(amip["model"])].reset_index()
    points = pd.concat([historical, amip], ignore_index=True)
    fig, ax, key_ax, visible = taylor_background(points, title, rmax=rmax, corr_min=corr_min,
                                                zoom=zoom, rms_levels=rms_levels, figsize=figsize)
    historical["in_view"], amip["in_view"] = visible[:len(historical)], visible[len(historical):]
    historical = historical.set_index("model")
    amip = amip.set_index("model")
    paired["in_view_historical"] = paired["model"].map(historical["in_view"])
    paired["in_view_amip"] = paired["model"].map(amip["in_view"])
    paired["arrow_drawn"] = False
    color_map, norm = taylor_bias_colors(points["bias"], bias_vlim, cmap)

    # Place the HISTORICAL tail first and the AMIP arrowhead second. A crop that
    # hides either endpoint omits that arrow, without connecting a different model.
    for index, row in paired.iterrows():
        h = historical.loc[row["model"]]
        a = amip.loc[row["model"]]
        if h["in_view"] and a["in_view"]:
            xy_h = (np.arccos(np.clip(h["correlation"], -1, 1)), h["std_ratio"])
            xy_a = (np.arccos(np.clip(a["correlation"], -1, 1)), a["std_ratio"])
            if not np.allclose(xy_h, xy_a, rtol=0, atol=1e-12):
                ax.annotate("", xy=xy_a, xytext=xy_h, xycoords="data", textcoords="data", zorder=3,
                            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0", color="0.35",
                                            linewidth=1.2, alpha=0.75, mutation_scale=12, shrinkA=6, shrinkB=6))
                paired.loc[index, "arrow_drawn"] = True

        # Equal fill rules for both experiments; outline color identifies experiment.
        # Each endpoint's fill uses its OWN signed mean-member bias.
        for data, edge in [(h, historical_edge), (a, amip_edge)]:
            if data["in_view"]:
                fill = color_map(norm(data["bias"])) if np.isfinite(data["bias"]) else "0.65"
                ax.scatter([np.arccos(np.clip(data["correlation"], -1, 1))], [data["std_ratio"]],
                           marker=model_markers[row["model"]], s=mean_size, color=fill,
                           edgecolors=edge, linewidths=1.8, zorder=5, clip_on=False)

    handles = [Line2D([], [], marker=model_markers[model], linestyle="none", markersize=8,
                      markerfacecolor="0.65", markeredgecolor="0.15", label=model) for model in paired["model"]]
    model_legend = key_ax.legend(handles=handles, title="Matched model shapes", loc="upper left",
                                frameon=False, fontsize=9, labelspacing=0.7)
    key_ax.add_artist(model_legend)
    edges = [Line2D([], [], marker="o", linestyle="none", markersize=9, markerfacecolor="0.8",
                    markeredgecolor=edge, markeredgewidth=1.8, label=label)
             for edge, label in [(historical_edge, "Historical: arrow tail"), (amip_edge, "AMIP: arrow head")]]
    key_ax.legend(handles=edges, loc="lower left", frameon=False, fontsize=9)
    add_taylor_colorbar(fig, color_map, norm, bias_label)
    return fig, ax, paired


# This is an add-on to the first cell of taylor_diagram_cells.ipynb. You do not
# need that notebook's old settings/application cells. These functions only
# organize existing scores: no maps, scores, or ensemble averages are recalculated.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings


def setup_taylor(metric, member_scores, mean_member_scores, *, bias_key="bias_over_std_x",
                 bias_label=None, bias_vlim=None, color_experiments=("amip_hist", "historical_matched"),
                 model_markers=None):
    """Return one reusable setup dictionary for one scored diagnostic.

    Uses the existing three-experiment prepare_taylor_tables function. Model
    summary coordinates are saved arithmetic means of MEMBER scores, not scores
    of averaged maps. The observations are the target x in every input score.
    Call again after changing/reloading scores; this dictionary is a snapshot.
    """
    experiments = ["amip_hist", "historical_matched", "historical"]
    labels = {
        "bias_over_std_x": r"Mean bias / obs spatial SD  $(\mu_y-\mu_x)/\sigma_x$",
        "bias_over_rms_x": r"Mean bias / obs RMS  $(\mu_y-\mu_x)/\mathrm{RMS}_x$",
        "bias": "Mean bias: model minus observations [diagnostic units]",
    }
    if bias_label is None:
        if bias_key not in labels:
            raise ValueError("Supply bias_label when choosing a different bias_key.")
        bias_label = labels[bias_key]

    # Check the requested diagnostic before gathering tables. A lone scalar
    # comparison has no spatial correlation/contrast and cannot supply a Taylor point.
    for experiment in experiments:
        if metric not in member_scores.get(experiment, {}) or metric not in mean_member_scores.get(experiment, {}):
            raise ValueError(f"{experiment}/{metric}: missing scores; calculate and score this diagnostic first.")
        names = list(member_scores[experiment][metric])
        if not names or set(names) != set(mean_member_scores[experiment][metric]):
            raise ValueError(f"{experiment}/{metric}: member and mean-score model sets must agree and be nonempty.")
        for model in names:
            for scores in [member_scores[experiment][metric][model], mean_member_scores[experiment][metric][model]]:
                if not {"pattern_correlation", "contrast_ratio", bias_key}.issubset(scores):
                    raise ValueError(f"{experiment}/{model}: need correlation, contrast_ratio and {bias_key}; "
                                     "single-scalar scores do not define Taylor geometry.")
    members, means = prepare_taylor_tables(member_scores, mean_member_scores, metric, bias_key=bias_key)
    amip_names = list(member_scores["amip_hist"][metric])
    if set(amip_names) != set(member_scores["historical_matched"][metric]):
        raise ValueError("Expected the same models in amip_hist and historical_matched.")

    # Keep a model's shape across experiments and diagnostic setups. Supplying
    # an earlier setup's markers also preserves shapes if dictionary order changes.
    markers = {} if model_markers is None else dict(model_markers)
    for experiment in experiments:
        names = list(member_scores[experiment][metric])
        if len(names) > len(TAYLOR_SHAPES):
            continue                       # Large groups use numbered circles instead.
        used = [repr(markers[name]) for name in names if name in markers]
        if len(used) != len(set(used)):
            raise ValueError(f"{experiment}: supplied model markers must be distinct within this group.")
        remaining = [shape for shape in TAYLOR_SHAPES if repr(shape) not in used]
        for name in names:
            if name not in markers:
                markers[name] = remaining.pop(0)

    # The default reproduces your matched-group scale, including both individual
    # members and model summaries. Include 'historical' here to cover its extremes
    # too, or supply bias_vlim manually to keep the scale fixed between diagnostics.
    color_experiments = [color_experiments] if isinstance(color_experiments, str) else list(color_experiments)
    if not color_experiments or not set(color_experiments).issubset(experiments):
        raise ValueError("color_experiments must select one or more of the three experiment names.")
    bias_values = pd.concat([table.loc[table.experiment.isin(color_experiments), "bias"]
                             for table in [members, means]]).to_numpy(dtype=float)
    _, norm = taylor_bias_colors(bias_values, bias_vlim=bias_vlim)
    for table in [members, means]:
        table.attrs.update(bias_key=bias_key, bias_label=bias_label, metric=metric)
    return {"metric": metric, "members": members, "means": means, "markers": markers,
            "bias_key": bias_key, "bias_label": bias_label, "bias_vlim": float(norm.vmax),
            "color_experiments": color_experiments}


def plot_taylor_setup(setup, experiment, **plot_options):
    """Short call to the original plotter; every plotting option can be overridden."""
    if experiment not in set(setup["means"]["experiment"]):
        raise ValueError(f"No experiment {experiment!r} in this setup.")
    options = {"title": f"{experiment} — {setup['metric']}", "bias_label": setup["bias_label"],
               "bias_vlim": setup["bias_vlim"], "large_group_bias": "color"}
    options.update(plot_options)
    return plot_taylor_experiment(setup["members"], setup["means"], experiment, setup["markers"], **options)

# Cell 2: arrows between independently chosen Taylor mean-score tables
def plot_taylor_transition(initial_means, final_means, model_markers=None, *,
                           initial_experiment=None, final_experiment=None,
                           initial_label=None, final_label=None, show_final_markers=True,
                           title=None, bias_label=None, bias_vlim=None, cmap="RdBu_r",
                           rmax=None, corr_min=None, zoom=False, rms_levels=None, mean_size=110,
                           initial_edge="0.15", final_edge="#009E73", arrow_color="0.35",
                           arrow_width=1.2, arrow_alpha=0.8, label_offsets=None, figsize=(14, 8.5)):
    """Connect matching MODEL mean-member-score points from two independent tables.

    Each table needs model, correlation, std_ratio and bias columns. Pass already
    selected one-experiment tables, or use initial_experiment/final_experiment.
    Diagnostics and experiments may differ; pairing always uses model names.
    No individual-member pairing or new averaging is performed.

    All scores should use comparable spatial weights/support. For different
    diagnostics, each point uses THAT diagnostic's own observational reference.
    Arrow length is not an RMSE between the two diagnostic maps. A mean-score
    point's contour value is also not the mean of member centered RMSE scores.
    """
    stages = []
    for table, experiment in [(initial_means, initial_experiment), (final_means, final_experiment)]:
        required = {"model", "correlation", "std_ratio", "bias"}
        if not required.issubset(table.columns):
            raise ValueError(f"Each endpoint table needs columns {sorted(required)}.")
        selected = table.copy()
        if experiment is not None:
            if "experiment" not in selected:
                raise ValueError("An experiment selector requires an 'experiment' column.")
            selected = selected.loc[selected.experiment == experiment].copy()
        if selected.empty or selected.model.isna().any() or selected.model.duplicated().any():
            raise ValueError("Each endpoint needs a nonempty table with exactly one row per named model.")
        for column in ["experiment", "metric"]:
            if column in selected and selected[column].nunique(dropna=False) != 1:
                raise ValueError(f"Select exactly one {column} for each endpoint.")
        stages.append(selected.reset_index(drop=True))
    initial, final = stages
    if set(initial.model) != set(final.model):
        raise ValueError("Endpoint model sets differ. Explicitly subset BOTH tables to the models you want.")

    # Use the INITIAL table's order for both stages, regardless of the final
    # table's original ordering. Never pair ensemble members by their row numbers.
    names = list(initial.model)
    final = final.set_index("model").loc[names].reset_index()
    if initial_label is None:
        initial_label = " / ".join(str(initial[column].iloc[0]) for column in ["experiment", "metric"]
                                    if column in initial) or "Initial"
    if final_label is None:
        final_label = " / ".join(str(final[column].iloc[0]) for column in ["experiment", "metric"]
                                  if column in final) or "Final"

    # setup_taylor attaches the chosen bias definition. Reject incompatible
    # definitions instead of silently sharing a misleading colorbar. For arbitrary
    # manually constructed tables, the caller supplies the interpretation/units.
    key_initial, key_final = initial_means.attrs.get("bias_key"), final_means.attrs.get("bias_key")
    if key_initial != key_final and key_initial is not None and key_final is not None:
        raise ValueError("Start/end bias_key must match to share a color scale.")
    metric_initial = initial["metric"].iloc[0] if "metric" in initial else None
    metric_final = final["metric"].iloc[0] if "metric" in final else None
    different_metrics = metric_initial is not None and metric_final is not None and metric_initial != metric_final
    if different_metrics and (key_initial == "bias" or key_final == "bias"):
        raise ValueError("For cross-diagnostic arrows, use normalized bias (bias_over_std_x or bias_over_rms_x).")
    if bias_label is None:
        if key_initial == key_final == "bias_over_std_x":
            bias_label = "Mean bias / each diagnostic's obs spatial SD"
        elif key_initial == key_final == "bias_over_rms_x":
            bias_label = "Mean bias / each diagnostic's obs RMS"
        else:
            bias_label = "Signed mean-member bias [input table units]"

    # Small groups retain filled model-specific shapes. Numbered initial circles
    # identify large groups, with bias still shown by COLOR, not by point size.
    markers = {} if model_markers is None else dict(model_markers)
    small_group = len(names) <= len(TAYLOR_SHAPES)
    if small_group:
        used = [repr(markers[name]) for name in names if name in markers]
        if len(used) != len(set(used)):
            raise ValueError("Supplied model markers must be distinct in the selected small group.")
        remaining = [shape for shape in TAYLOR_SHAPES if repr(shape) not in used]
        for name in names:
            if name not in markers:
                markers[name] = remaining.pop(0)
    if not np.isfinite(mean_size) or mean_size <= 0:
        raise ValueError("mean_size must be positive and finite.")

    # BOTH sets of coordinates determine the extent even when final markers are
    # hidden. The color scale also spans both sets of biases, so switching final
    # markers on/off does not recolor the initial points. Override bias_vlim to
    # match another figure exactly; setup dictionaries are otherwise independent.
    points = pd.concat([initial, final], ignore_index=True)
    color_map, norm = taylor_bias_colors(points.bias, bias_vlim=bias_vlim, cmap=cmap)
    fig, ax, key_ax, visible = taylor_background(points, title or f"{initial_label} → {final_label}",
                                                rmax=rmax, corr_min=corr_min, zoom=zoom,
                                                rms_levels=rms_levels, figsize=figsize)
    # Reserve additional room for the cross-diagnostic note and the colorbar
    # label. Adjust the axes before placing any model-number annotations.
    ax.get_subplotspec().get_gridspec().update(bottom=.28)
    initial["in_view"], final["in_view"] = visible[:len(names)], visible[len(names):]
    paired = initial.merge(final, on="model", suffixes=("_initial", "_final"), validate="one_to_one")
    paired["number"], paired["arrow_drawn"] = np.arange(1, len(names) + 1), False
    paired["initial_marker_drawn"] = paired.in_view_initial
    paired["final_marker_drawn"] = paired.in_view_final & bool(show_final_markers)
    ti, ri, _ = taylor_positions(initial)
    tf, rf, _ = taylor_positions(final)

    # Arrows are straight in the displayed Cartesian Taylor plane. Their data
    # endpoints remain exact (zero shrink); an optional final marker covers the
    # arrow tip. Undefined or cropped endpoints suppress the connecting arrow,
    # but any visible initial/final marker can still be shown independently.
    for i, model in enumerate(names):
        xy_initial, xy_final = (ti[i], ri[i]), (tf[i], rf[i])
        cart_initial = np.array([ri[i] * np.cos(ti[i]), ri[i] * np.sin(ti[i])])
        cart_final = np.array([rf[i] * np.cos(tf[i]), rf[i] * np.sin(tf[i])])
        both_visible = initial.in_view.iloc[i] and final.in_view.iloc[i]
        if both_visible and not np.allclose(cart_initial, cart_final, rtol=0, atol=1e-12):
            ax.annotate("", xy=xy_final, xytext=xy_initial, xycoords="data", textcoords="data", zorder=3,
                        arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0", color=arrow_color,
                                        linewidth=arrow_width, alpha=arrow_alpha, mutation_scale=12,
                                        shrinkA=0, shrinkB=0))
            paired.loc[i, "arrow_drawn"] = True
        for table, edge, show, zorder in [(initial, initial_edge, True, 5),
                                         (final, final_edge, show_final_markers, 4)]:
            if show and table.in_view.iloc[i]:
                row = table.iloc[i]
                fill = color_map(norm(row.bias)) if np.isfinite(row.bias) else "0.65"
                ax.scatter(np.arccos(np.clip(row.correlation, -1, 1)), row.std_ratio,
                           marker=markers[model] if small_group else "o", s=mean_size, color=fill,
                           edgecolors=edge, linewidths=1.8, zorder=zorder, clip_on=False)

    # Keep the model key outside the diagram. For large groups, numbers label
    # initial points only; arrows (and optional endpoint symbols) identify finals.
    if small_group:
        handles = [Line2D([], [], marker=markers[model], linestyle="none", markersize=8,
                          markerfacecolor="0.65", markeredgecolor=initial_edge, label=model) for model in names]
        legend = key_ax.legend(handles=handles, title="Matched model shapes", loc="upper left",
                               frameon=False, fontsize=9, labelspacing=0.7)
        key_ax.add_artist(legend)
    else:
        numbered = initial.assign(number=paired.number.to_numpy(), marker_area=mean_size)
        label_taylor_models(ax, numbered.loc[numbered.in_view], {} if label_offsets is None else label_offsets)
        columns = 1 if len(names) <= 24 else 2
        rows = int(np.ceil(len(names) / columns))
        key_ax.text(0, 1, "Model numbers at initial points", fontsize=10, va="top")
        for i, model in enumerate(names):
            column, line = divmod(i, rows)
            key_ax.text(column * .54, .93 - line * .72 / rows, f"{i+1:02d}  {model}", fontsize=8, va="top")
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=9, markerfacecolor="0.8",
                      markeredgecolor=initial_edge, markeredgewidth=1.8, label="Initial: arrow tail")]
    if show_final_markers:
        handles.append(Line2D([], [], marker="o", linestyle="none", markersize=9, markerfacecolor="0.8",
                               markeredgecolor=final_edge, markeredgewidth=1.8, label="Final: arrow head"))
    else:
        handles.append(Line2D([], [], color=arrow_color, linewidth=arrow_width, label="Final: unmarked arrow head"))
    key_ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
    colorbar = add_taylor_colorbar(fig, color_map, norm, bias_label)
    colorbar.ax.set_position([.11, .155, .43, .022])
    fig.text(.06, .072, "Each endpoint uses its own diagnostic's OBS reference. "
             "Arrow length is not an error between endpoint maps.", fontsize=9, color="0.3")
    paired.attrs.update(initial_label=initial_label, final_label=final_label, bias_vlim=float(norm.vmax),
                        bias_key=key_initial if key_initial == key_final else None)
    return fig, ax, paired