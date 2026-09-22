"""Visualization and widget helpers for CSC320 notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Math, display
from PIL import Image
from ipywidgets import (
    Button,
    FloatSlider,
    HBox,
    Layout,
    ToggleButtons,
    VBox,
    interactive_output,
)

from geometry import (
    apply_homography,
    camera_project,
    estimate_affine_transform,
    estimate_homography,
    estimate_horizon,
    estimate_vanishing_point,
    jitter_distribution_diagnostics,
    line_from_points,
    make_affine_transform,
    rectangle_geometry_metrics,
    rectification_baseline_results,
    rectification_jitter_experiment,
    rectification_target,
    rectify_plane,
    warp_image,
)


IMAGE_DIR = Path("images")


def load_image(name, image_dir=IMAGE_DIR):
    """Load an RGB image from the course image directory."""
    path = Path(image_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"Add a real photo at {path}")
    return np.asarray(Image.open(path).convert("RGB"))


def show(img, title=None, ax=None):
    """Display an image on a matplotlib axis."""
    ax = ax or plt.gca()
    ax.imshow(img, cmap="gray" if np.asarray(img).ndim == 2 else None)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


def _fit_notebook_width(fig, max_width_inches=10.5):
    """Keep a wide Matplotlib canvas inside the notebook output column."""
    width, height = fig.get_size_inches()
    if width > max_width_inches:
        fig.set_size_inches(max_width_inches, height, forward=True)
    layout = getattr(fig.canvas, "layout", None)
    if layout is not None:
        layout.width = "100%"
        layout.max_width = "100%"
    return fig


def checkerboard(size=256, squares=16):
    """Create a synthetic checkerboard image."""
    y, x = np.indices((size, size))
    board = ((x // (size // squares) + y // (size // squares)) % 2) * 255
    return board.astype(np.uint8)


def matrix_latex(H, name):
    """Format a numeric matrix as a LaTeX bmatrix."""
    rows = []
    for row in np.asarray(H):
        rows.append(" & ".join(f"{value: .2f}" for value in row))
    return rf"{name}=\begin{{bmatrix}}" + r"\\".join(rows) + r"\end{bmatrix}"


def normalize_homography_for_display(H, source_shape, destination_shape):
    """Represent a pixel homography using [0, 1] source and destination coordinates."""
    src_h, src_w = source_shape[:2]
    dst_h, dst_w = destination_shape[:2]
    source_to_unit = np.array(
        [[1 / (src_w - 1), 0, 0], [0, 1 / (src_h - 1), 0], [0, 0, 1]],
        dtype=float,
    )
    dest_to_unit = np.array(
        [[1 / (dst_w - 1), 0, 0], [0, 1 / (dst_h - 1), 0], [0, 0, 1]],
        dtype=float,
    )
    H_unit = dest_to_unit @ H @ np.linalg.inv(source_to_unit)
    return H_unit / H_unit[-1, -1]


def affine_match_widget(img, target_img):
    """Display affine sliders next to a fixed target image."""
    controls = {
        "scale_x": FloatSlider(description="scale_x", min=0.35, max=1.75, step=0.01, value=1.0, readout_format=".2f"),
        "scale_y": FloatSlider(description="scale_y", min=0.35, max=1.75, step=0.01, value=1.0, readout_format=".2f"),
        "rotation": FloatSlider(description="rotation", min=-60, max=60, step=1, value=0, readout_format=".0f"),
        "shear_x": FloatSlider(description="shear_x", min=-1.20, max=1.20, step=0.01, value=0.0, readout_format=".2f"),
        "shear_y": FloatSlider(description="shear_y", min=-1.20, max=1.20, step=0.01, value=0.0, readout_format=".2f"),
        "tx": FloatSlider(description="tx", min=-120, max=120, step=1, value=0, readout_format=".0f"),
        "ty": FloatSlider(description="ty", min=-120, max=120, step=1, value=0, readout_format=".0f"),
    }

    def draw(scale_x, scale_y, rotation, shear_x, shear_y, tx, ty):
        H_current = make_affine_transform(
            scale=(scale_x, scale_y),
            rotation_degrees=rotation,
            shear=(shear_x, shear_y),
            translation=(tx, ty),
        )
        current_img = warp_image(img, H_current, output_shape=target_img.shape[:2])

        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        show(target_img, "Fixed target", ax[0])
        show(current_img, "Your sliders", ax[1])
        plt.tight_layout()
        plt.show()

    left_controls = VBox([controls["scale_x"], controls["scale_y"], controls["rotation"]])
    right_controls = VBox([controls["shear_x"], controls["shear_y"], controls["tx"], controls["ty"]])
    ui = HBox([left_controls, right_controls], layout=Layout(gap="24px"))
    out = interactive_output(draw, controls)
    display(ui, out)
    return controls


def affine_slider_params(controls):
    """Read affine parameters from the affine widget controls."""
    return {
        "scale": (controls["scale_x"].value, controls["scale_y"].value),
        "rotation_degrees": controls["rotation"].value,
        "shear": (controls["shear_x"].value, controls["shear_y"].value),
        "translation": (controls["tx"].value, controls["ty"].value),
    }


def draw_projected_posts(focal_length=700, pitch=-8, yaw=0, roll=0, camera_x=0, camera_y=0, camera_z=0, ax=None, title="Pinhole projection"):
    """Draw equal-height poles standing on a projected ground plane."""
    image_center = (320, 260)
    rotation = (pitch, yaw, roll)
    camera_position = (camera_x, camera_y, camera_z)
    pole_height = 1.8
    poles = [
        ((-1.6, 0.0, 5.5), "tab:blue"),
        ((2.1, 0.0, 9.0), "tab:red"),
        ((-0.2, 0.0, 7.0), "0.45"),
        ((1.1, 0.0, 10.5), "0.45"),
    ]
    x_min, x_max = -2.7, 2.7
    z_min, z_max = 4.4, 11.4

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    def project(points):
        return camera_project(
            points,
            focal_length=focal_length,
            image_center=image_center,
            rotation_degrees=rotation,
            camera_position=camera_position,
        )

    plane_corners_3d = np.array(
        [[x_min, 0, z_min], [x_max, 0, z_min], [x_max, 0, z_max], [x_min, 0, z_max]],
        dtype=float,
    )
    plane_corners, plane_depths = project(plane_corners_3d)
    if np.all(plane_depths > 0) and np.all(np.isfinite(plane_corners)):
        ax.fill(plane_corners[:, 0], plane_corners[:, 1], color="lightgray", alpha=0.35, zorder=0)
        ax.plot(*np.vstack([plane_corners, plane_corners[0]]).T, color="0.35", linewidth=2)

    for x in np.linspace(x_min, x_max, 6):
        line_3d = np.array([[x, 0, z_min], [x, 0, z_max]], dtype=float)
        line_2d, depths = project(line_3d)
        if np.all(depths > 0) and np.all(np.isfinite(line_2d)):
            ax.plot(line_2d[:, 0], line_2d[:, 1], color="0.65", linewidth=1)
    for z in np.linspace(z_min, z_max, 6):
        line_3d = np.array([[x_min, 0, z], [x_max, 0, z]], dtype=float)
        line_2d, depths = project(line_3d)
        if np.all(depths > 0) and np.all(np.isfinite(line_2d)):
            ax.plot(line_2d[:, 0], line_2d[:, 1], color="0.65", linewidth=1)

    bases = []
    for base, color in poles:
        base = np.asarray(base, dtype=float)
        top = base + np.array([0, -pole_height, 0], dtype=float)
        segment, depths = project(np.vstack([base, top]))
        bases.append(base)
        if np.all(depths > 0) and np.all(np.isfinite(segment)):
            ax.plot(segment[:, 0], segment[:, 1], color=color, marker="o", linewidth=3)

    base_pts, base_depths = project(np.asarray(bases))
    valid = (base_depths > 0) & np.all(np.isfinite(base_pts), axis=1)
    ax.scatter(base_pts[valid, 0], base_pts[valid, 1], s=18, c="black", zorder=3)
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("image x")
    ax.set_ylabel("image y")
    return fig, ax


def projection_widget():
    """Display a simple interactive pinhole projection widget."""
    controls = {
        "focal_length": FloatSlider(min=250, max=1600, step=25, value=700, description="focal", readout_format=".0f"),
        "pitch": FloatSlider(min=-45, max=45, step=1, value=-8, description="pitch", readout_format=".0f"),
        "yaw": FloatSlider(min=-70, max=70, step=1, value=0, description="yaw", readout_format=".0f"),
        "roll": FloatSlider(min=-90, max=90, step=1, value=0, description="roll", readout_format=".0f"),
        "camera_x": FloatSlider(min=-4.0, max=4.0, step=0.1, value=0, description="cam x", readout_format=".1f"),
        "camera_y": FloatSlider(min=-3.0, max=3.0, step=0.1, value=0, description="cam y", readout_format=".1f"),
        "camera_z": FloatSlider(min=-2.0, max=4.0, step=0.1, value=0, description="cam z", readout_format=".1f"),
    }
    out = interactive_output(draw_projected_posts, controls)
    display(VBox(list(controls.values())), out)
    return controls


def plane_rectification_tool(image_name="poster.jpg", figsize=(11, 5), image_dir=IMAGE_DIR):
    """Click four plane corners, then drag them with a live rectified preview."""
    poster = load_image(image_name, image_dir=image_dir)
    corners = []
    selected_corner = {"index": None}
    state = {"image": poster, "corners": corners, "H_rect": None, "rectified": None}

    fig, (ax_image, ax_rectified) = plt.subplots(1, 2, figsize=figsize)
    show(poster, "Click 4 corners, then drag to refine", ax_image)
    ax_rectified.set_title("Rectified plane")
    ax_rectified.set_axis_off()
    rectified_artist = ax_rectified.imshow(np.zeros((10, 10, 3), dtype=np.uint8))
    placeholder = ax_rectified.text(
        0.5,
        0.5,
        "waiting for 4 corners",
        ha="center",
        va="center",
        transform=ax_rectified.transAxes,
    )
    corner_points, = ax_image.plot([], [], "yo", markersize=7, markeredgecolor="black")
    corner_lines, = ax_image.plot([], [], "y-", linewidth=2)

    def update_overlay():
        pts = np.asarray(corners, dtype=float)
        if len(pts) == 0:
            corner_points.set_data([], [])
            corner_lines.set_data([], [])
        else:
            corner_points.set_data(pts[:, 0], pts[:, 1])
            if len(pts) >= 2:
                line_pts = pts if len(pts) < 4 else np.vstack([pts, pts[0]])
                corner_lines.set_data(line_pts[:, 0], line_pts[:, 1])
        if len(corners) == 4:
            update_rectified_view()
        fig.canvas.draw_idle()

    def update_rectified_view():
        pts = np.asarray(corners, dtype=float)
        rectified, H_rect = rectify_plane(poster, pts)
        state["H_rect"] = H_rect
        state["rectified"] = rectified
        rectified_artist.set_data(rectified)
        rectified_artist.set_extent((0, rectified.shape[1], rectified.shape[0], 0))
        ax_rectified.set_xlim(0, rectified.shape[1])
        ax_rectified.set_ylim(rectified.shape[0], 0)
        placeholder.set_visible(False)

    def nearest_corner(event, max_distance_pixels=12):
        if len(corners) == 0:
            return None
        pts = np.asarray(corners, dtype=float)
        display_pts = ax_image.transData.transform(pts)
        event_pt = np.array([event.x, event.y], dtype=float)
        distances = np.linalg.norm(display_pts - event_pt, axis=1)
        index = int(np.argmin(distances))
        if distances[index] <= max_distance_pixels:
            return index
        return None

    def on_click(event):
        if event.inaxes != ax_image or event.xdata is None or event.ydata is None:
            return
        if len(corners) == 4:
            selected_corner["index"] = nearest_corner(event)
            return
        corners.append([event.xdata, event.ydata])
        update_overlay()

    def on_drag(event):
        index = selected_corner["index"]
        if index is None or event.inaxes != ax_image or event.xdata is None or event.ydata is None:
            return
        corners[index] = [event.xdata, event.ydata]
        update_overlay()

    def on_release(_):
        selected_corner["index"] = None

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", on_drag)
    fig.canvas.mpl_connect("button_release_event", on_release)
    state["figure"] = fig
    state["image_axis"] = ax_image
    state["rectified_axis"] = ax_rectified
    return state


def affine_homography_comparison_tool(images, aspect_ratio, figsize=(12, 8)):
    """Collect outer and held-out corners for two views on one canvas.

    Parameters
    ----------
    images : mapping[str, ndarray]
        Named views of the same planar rectangular object.
    aspect_ratio : float
        Physical object width divided by physical object height.
    figsize : tuple
        Matplotlib figure size.

    Returns
    -------
    state : dict
        Live selections, transforms, rectified images, and held-out metrics.
    """
    images = {str(name): np.asarray(image) for name, image in images.items()}
    if len(images) < 2:
        raise ValueError("Provide at least two named views")
    if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be finite and positive")

    destination, output_shape = rectification_target(aspect_ratio)
    output_height, output_width = output_shape

    views = {
        name: {
            "image": image,
            "outer": [],
            "inner": [],
            "destination_corners": destination.copy(),
            "H_affine": None,
            "H_homography": None,
            "affine_rectified": None,
            "homography_rectified": None,
            "metrics": None,
        }
        for name, image in images.items()
    }
    first_name = next(iter(images))
    state = {
        "active_view": first_name,
        "active_set": "outer",
        "views": views,
        "output_shape": (output_height, output_width),
        "aspect_ratio": float(aspect_ratio),
    }

    fig = plt.figure(figsize=figsize)
    axes = fig.subplot_mosaic(
        [["source", "source"], ["affine", "homography"]],
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )
    _fit_notebook_width(fig)
    ax_source = axes["source"]
    ax_affine = axes["affine"]
    ax_homography = axes["homography"]
    view_toggle = ToggleButtons(
        options=[(name.replace("_", " ").title(), name) for name in images],
        value=first_name,
        description="Photo:",
        layout=Layout(width="auto"),
    )
    set_toggle = ToggleButtons(
        options=[("Outer fit corners", "outer"), ("Held-out inner rectangle", "inner")],
        value="outer",
        description="Selecting:",
        layout=Layout(width="auto"),
    )
    undo_button = Button(description="Undo active point")
    clear_button = Button(description="Clear active points")
    status = ax_source.text(
        0.02,
        0.98,
        "",
        transform=ax_source.transAxes,
        color="white",
        va="top",
        bbox={"facecolor": "black", "alpha": 0.7, "pad": 4},
    )

    def draw_polygon(axis, points, colour, label):
        if points is None or len(points) == 0:
            return
        points = np.asarray(points, dtype=float)
        axis.plot(points[:, 0], points[:, 1], "o", color=colour, markersize=6)
        if len(points) >= 2:
            displayed = points if len(points) < 4 else np.vstack([points, points[0]])
            axis.plot(displayed[:, 0], displayed[:, 1], "-", color=colour, linewidth=2)
        for index, point in enumerate(points):
            axis.annotate(
                str(index + 1),
                point,
                xytext=(4, 4),
                textcoords="offset points",
                color=colour,
                fontsize=9,
                weight="bold",
            )
        axis.plot([], [], "-", color=colour, label=label)

    def update_view_results(view):
        if len(view["outer"]) != 4:
            view["H_affine"] = None
            view["H_homography"] = None
            view["affine_rectified"] = None
            view["homography_rectified"] = None
            view["metrics"] = None
            return

        outer = np.asarray(view["outer"], dtype=float)
        H_affine = estimate_affine_transform(outer, destination)
        H_homography = estimate_homography(outer, destination)
        view["H_affine"] = H_affine
        view["H_homography"] = H_homography
        view["affine_rectified"] = warp_image(
            view["image"], H_affine, state["output_shape"]
        )
        view["homography_rectified"] = warp_image(
            view["image"], H_homography, state["output_shape"]
        )

        if len(view["inner"]) == 4:
            inner = np.asarray(view["inner"], dtype=float)
            affine_inner = apply_homography(H_affine, inner)
            homography_inner = apply_homography(H_homography, inner)
            view["metrics"] = {
                "affine": rectangle_geometry_metrics(affine_inner),
                "homography": rectangle_geometry_metrics(homography_inner),
                "affine_inner": affine_inner,
                "homography_inner": homography_inner,
            }
        else:
            view["metrics"] = None

    def draw_result(axis, image, title, transformed_inner=None, metrics=None):
        axis.clear()
        if image is None:
            axis.set_title(title)
            axis.text(0.5, 0.5, "select 4 outer corners", ha="center", va="center")
            axis.set_axis_off()
            return
        show(image, title, axis)
        if transformed_inner is not None:
            draw_polygon(axis, transformed_inner, "tab:cyan", "held-out rectangle")
            axis.legend(loc="lower right", fontsize=8)
        if metrics is not None:
            axis.set_title(
                f"{title}\nright-angle error {metrics['right_angle_error_deg']:.2f}°; "
                f"parallel error {metrics['parallel_error_deg']:.2f}°"
            )

    def redraw():
        view = views[state["active_view"]]
        update_view_results(view)

        ax_source.clear()
        show(view["image"], f"{state['active_view']}: source", ax_source)
        draw_polygon(ax_source, view["outer"], "tab:orange", "outer fit corners")
        draw_polygon(ax_source, view["inner"], "tab:cyan", "held-out rectangle")
        selected = view[state["active_set"]]
        status.set_text(
            f"{state['active_set']}: {len(selected)}/4 points | "
            "top-left, then clockwise"
        )
        ax_source.add_artist(status)
        if view["outer"] or view["inner"]:
            ax_source.legend(loc="lower right", fontsize=8)

        metrics = view["metrics"]
        draw_result(
            ax_affine,
            view["affine_rectified"],
            "Affine least-squares fit",
            None if metrics is None else metrics["affine_inner"],
            None if metrics is None else metrics["affine"],
        )
        draw_result(
            ax_homography,
            view["homography_rectified"],
            "Homography fit",
            None if metrics is None else metrics["homography_inner"],
            None if metrics is None else metrics["homography"],
        )
        fig.tight_layout()
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax_source or event.xdata is None or event.ydata is None:
            return
        active_points = views[state["active_view"]][state["active_set"]]
        if len(active_points) < 4:
            active_points.append([event.xdata, event.ydata])
            redraw()

    def change_view(change):
        if change.get("name") == "value":
            state["active_view"] = change["new"]
            redraw()

    def change_set(change):
        if change.get("name") == "value":
            state["active_set"] = change["new"]
            redraw()

    def undo_active(_):
        points = views[state["active_view"]][state["active_set"]]
        if points:
            points.pop()
        redraw()

    def clear_active(_):
        views[state["active_view"]][state["active_set"]].clear()
        redraw()

    view_toggle.observe(change_view, names="value")
    set_toggle.observe(change_set, names="value")
    undo_button.on_click(undo_active)
    clear_button.on_click(clear_active)
    fig.canvas.mpl_connect("button_press_event", on_click)
    display(VBox([HBox([view_toggle, set_toggle]), HBox([undo_button, clear_button])]))
    state["figure"] = fig
    state["refresh"] = redraw
    redraw()
    plt.show()
    return state


def show_rectification_photo_set(photo_data, figsize=None):
    """Print the capture audit and display all required photographs."""
    if photo_data["using_demo_data"]:
        print("DEMO DATA ONLY. Missing:", ", ".join(photo_data["missing_files"]))
        print("The synthetic fallback demonstrates the tool but does not satisfy Task 1.")
    elif photo_data["submission_data_pass"]:
        print(
            "Submission data check: PASS. All required photographs meet the "
            "resolution requirement."
        )
    else:
        print(
            "Submission data check: NOT READY. Recapture undersized photographs:",
            ", ".join(photo_data["undersized"]),
        )

    if figsize is None:
        figsize = (5 * len(photo_data["photos"]), 5)
    fig, axes = plt.subplots(1, len(photo_data["photos"]), figsize=figsize)
    axes = np.atleast_1d(axes)
    for axis, (name, image) in zip(axes, photo_data["photos"].items()):
        show(image, f"{name} view: loaded {image.shape[1]} × {image.shape[0]}", axis)
    plt.tight_layout()
    plt.show()
    return fig, axes


show_rectification_photo_pair = show_rectification_photo_set


def show_rectification_baseline(comparison_state, figsize=None):
    """Print and visualize held-out affine/homography baseline measurements."""
    rows = rectification_baseline_results(comparison_state)
    print(f"{'view':10s} {'model':12s} {'right-angle error':>19s} {'parallel error':>17s}")
    for row in rows:
        print(
            f"{row['view']:10s} {row['model']:12s} "
            f"{row['right_angle_error_deg']:19.3f} "
            f"{row['parallel_error_deg']:17.3f}"
        )

    if figsize is None:
        figsize = (14, 4.25 * len(comparison_state["views"]))
    fig, axes = plt.subplots(len(comparison_state["views"]), 3, figsize=figsize)
    _fit_notebook_width(fig)
    axes = np.atleast_2d(axes)
    for row_axes, (view_name, view) in zip(axes, comparison_state["views"].items()):
        show(view["image"], f"{view_name}: selected source", row_axes[0])
        for point_set, colour in (("outer", "tab:orange"), ("inner", "tab:cyan")):
            points = np.asarray(view[point_set], dtype=float)
            closed = np.vstack([points, points[0]])
            row_axes[0].plot(closed[:, 0], closed[:, 1], "-o", color=colour, linewidth=2)
        show(view["affine_rectified"], f"{view_name}: affine", row_axes[1])
        show(view["homography_rectified"], f"{view_name}: homography", row_axes[2])
        for axis, model in zip(row_axes[1:], ("affine", "homography")):
            points = view["metrics"][f"{model}_inner"]
            closed = np.vstack([points, points[0]])
            axis.plot(closed[:, 0], closed[:, 1], "-o", color="tab:cyan", linewidth=2)
    plt.tight_layout()
    plt.show()
    return rows


def show_jitter_diagnostic(
    comparison_state,
    jitter_function,
    view_name="strong",
    sigma_px=8,
    trials=400,
    seed=320,
    figsize=(9, 5),
):
    """Validate and visualize samples produced by a student jitter function."""
    if view_name not in comparison_state["views"]:
        raise KeyError(f"Unknown comparison view: {view_name}")
    view = comparison_state["views"][view_name]
    points = np.asarray(view["outer"], dtype=float)
    if points.shape != (4, 2):
        raise RuntimeError(f"Select all four outer corners for the {view_name} view first")
    diagnostic = jitter_distribution_diagnostics(
        points,
        jitter_function,
        sigma_px=sigma_px,
        trials=trials,
        seed=seed,
    )

    print(f"Requested sigma: {diagnostic['sigma_px']:.2f} px")
    print(
        "Empirical mean offset [dx, dy]: "
        f"{diagnostic['empirical_mean'].round(3)} px"
    )
    print(
        "Empirical standard deviation [dx, dy]: "
        f"{diagnostic['empirical_std'].round(3)} px"
    )
    print(
        "Largest absolute cross-coordinate correlation: "
        f"{diagnostic['largest_cross_correlation']:.3f}"
    )
    if diagnostic["plausible"]:
        print("Distribution check: plausible independent zero-mean Gaussian jitter.")
    else:
        print("WARNING: the distribution does not resemble independent zero-mean Gaussian jitter.")

    samples = diagnostic["samples"]
    offsets = diagnostic["offsets"]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _fit_notebook_width(fig)
    show(view["image"], "Jittered outer quadrilaterals", axes[0])
    for sample in samples[:60]:
        closed = np.vstack([sample, sample[0]])
        axes[0].plot(closed[:, 0], closed[:, 1], color="tab:orange", alpha=0.06)
    corner_colours = ("tab:red", "tab:green", "tab:blue", "tab:purple")
    for corner_index, colour in enumerate(corner_colours):
        cloud = samples[:, corner_index]
        axes[0].scatter(cloud[:, 0], cloud[:, 1], s=6, color=colour, alpha=0.12)
        axes[0].plot(
            points[corner_index, 0],
            points[corner_index, 1],
            marker="x",
            color=colour,
            markersize=10,
            markeredgewidth=2,
        )
        corner_offsets = offsets[:, corner_index]
        axes[1].scatter(
            corner_offsets[:, 0],
            corner_offsets[:, 1],
            s=8,
            color=colour,
            alpha=0.18,
            label=f"corner {corner_index + 1}",
        )

    axes[1].axhline(0, color="black", linewidth=1, alpha=0.5)
    axes[1].axvline(0, color="black", linewidth=1, alpha=0.5)
    axes[1].add_patch(
        plt.Circle(
            (0, 0),
            diagnostic["sigma_px"],
            fill=False,
            linestyle="--",
            color="black",
            alpha=0.6,
            label="radius = sigma reference",
        )
    )
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("x offset (px)")
    axes[1].set_ylabel("y offset (px)")
    axes[1].set_title("Distribution of coordinate offsets")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
    return diagnostic


def show_rectification_jitter_experiment(
    comparison_state,
    jitter_function,
    noise_levels=(0, 2, 5, 10),
    trials=40,
    seed=320,
    figsize=(9, 4.5),
    view_names=None,
):
    """Run, print, and plot the rectification corner-jitter experiment."""
    experiment = rectification_jitter_experiment(
        comparison_state,
        jitter_function,
        noise_levels=noise_levels,
        trials=trials,
        seed=seed,
        view_names=view_names,
    )
    summary = experiment["summary"]
    plotted_views = list(dict.fromkeys(row["view"] for row in summary))
    print(f"{'view':10s} {'model':12s} {'noise px':>8s} {'median':>10s} {'IQR':>19s}")
    for row in summary:
        print(
            f"{row['view']:10s} {row['model']:12s} {row['noise_px']:8.1f} "
            f"{row['median']:10.3f} [{row['q25']:.3f}, {row['q75']:.3f}]"
        )

    fig, axes = plt.subplots(
        1, len(plotted_views), figsize=figsize, sharey=True
    )
    _fit_notebook_width(fig)
    axes = np.atleast_1d(axes)
    for axis, view_name in zip(axes, plotted_views):
        for model, colour in (("affine", "tab:orange"), ("homography", "tab:blue")):
            rows = [
                row
                for row in summary
                if row["view"] == view_name and row["model"] == model
            ]
            x = np.array([row["noise_px"] for row in rows])
            median = np.array([row["median"] for row in rows])
            lower = np.array([row["q25"] for row in rows])
            upper = np.array([row["q75"] for row in rows])
            axis.plot(x, median, "-o", color=colour, label=model)
            axis.fill_between(x, lower, upper, color=colour, alpha=0.15)
        axis.set_title(f"{view_name} view")
        axis.set_xlabel("outer-corner noise standard deviation (px)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("held-out mean right-angle error (degrees)")
    axes[-1].legend()
    plt.tight_layout()
    plt.show()
    return experiment


def vanishing_point_repeat_click_tool(image, repeats=1, figsize=(10, 7)):
    """Collect two-point line clicks for two vanishing points and show the horizon."""
    state = {
        "active": "A",
        "pending": [],
        "lines": {"A": [], "B": []},
        "vanishing_points": {"A": None, "B": None},
        "horizon": None,
    }

    fig, ax = plt.subplots(figsize=figsize)
    show(image, "Click two endpoints per line", ax)
    status = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        color="white",
        va="top",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 4},
    )

    vp_button = Button(description="Vanishing point: A")
    new_line_button = Button(description="New line")
    clear_button = Button(description="Clear")

    colors = {"A": "tab:cyan", "B": "tab:orange"}

    def set_status():
        status.set_text(
            f"Vanishing point {state['active']} | click endpoint 1 then endpoint 2"
        )

    def draw_infinite_line(line, color, alpha=0.18, linewidth=1):
        a, b, c = line
        x_min, x_max = ax.get_xlim()
        y_max, y_min = ax.get_ylim()
        candidates = []
        if abs(b) > 1e-12:
            for x in [x_min, x_max]:
                y = -(a * x + c) / b
                if y_min <= y <= y_max:
                    candidates.append((x, y))
        if abs(a) > 1e-12:
            for y in [y_min, y_max]:
                x = -(b * y + c) / a
                if x_min <= x <= x_max:
                    candidates.append((x, y))
        if len(candidates) >= 2:
            p1, p2 = np.asarray(candidates[:2], dtype=float)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=alpha, linewidth=linewidth)

    def redraw():
        ax.clear()
        show(image, "Repeated clicks reveal line and vanishing-point stability", ax)
        ax.add_artist(status)

        for vp_name, segments in state["lines"].items():
            color = colors[vp_name]
            for seg in segments:
                ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=3)
                draw_infinite_line(line_from_points(seg[0], seg[1]), color, alpha=0.35, linewidth=1.5)

            if len(segments) >= 2:
                lines = np.array([line_from_points(seg[0], seg[1]) for seg in segments])
                vp = estimate_vanishing_point(lines)
                state["vanishing_points"][vp_name] = vp
                ax.plot(vp[0], vp[1], "x", color=color, markersize=12, markeredgewidth=3)

        vp_a = state["vanishing_points"]["A"]
        vp_b = state["vanishing_points"]["B"]
        if vp_a is not None and vp_b is not None:
            horizon = estimate_horizon(vp_a, vp_b)
            state["horizon"] = horizon
            draw_infinite_line(horizon, "red", alpha=0.9, linewidth=2.5)

        if state["pending"]:
            pts = np.asarray(state["pending"], dtype=float)
            ax.plot(pts[:, 0], pts[:, 1], "o", color=colors[state["active"]])

        set_status()
        fig.canvas.draw_idle()

    def finish_pending_line():
        if len(state["pending"]) == 2:
            arr = np.asarray(state["pending"], dtype=float).reshape(2, 2)
            state["lines"][state["active"]].append(arr)
            state["pending"] = []
            redraw()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        state["pending"].append([event.xdata, event.ydata])
        finish_pending_line()
        redraw()

    def toggle_vp(_):
        state["active"] = "B" if state["active"] == "A" else "A"
        vp_button.description = f"Vanishing point: {state['active']}"
        state["pending"] = []
        redraw()

    def new_line(_):
        state["pending"] = []
        redraw()

    def clear(_):
        state["pending"] = []
        state["lines"] = {"A": [], "B": []}
        state["vanishing_points"] = {"A": None, "B": None}
        state["horizon"] = None
        redraw()

    vp_button.on_click(toggle_vp)
    new_line_button.on_click(new_line)
    clear_button.on_click(clear)
    fig.canvas.mpl_connect("button_press_event", on_click)
    display(HBox([vp_button, new_line_button, clear_button]))
    redraw()
    return state


def vanishing_directions_click_tool(image, directions=None, figsize=(10, 7)):
    """Select several vanishing directions on one interactive canvas.

    Parameters
    ----------
    image : ndarray
        Image on which students click line endpoints.
    directions : sequence of dict, optional
        Each dictionary must contain ``key``, ``label``, and ``color``. The
        selected segments and fitted vanishing point for every direction are
        stored in the returned state's ``directions`` dictionary.
    figsize : tuple
        Matplotlib figure size.
    """
    image = np.asarray(image)
    height, width = image.shape[:2]
    if directions is None:
        directions = [
            {
                "key": "facade_a",
                "label": "Horizontal lines: left facade",
                "color": "tab:cyan",
            },
            {
                "key": "facade_b",
                "label": "Horizontal lines: right facade",
                "color": "tab:orange",
            },
        ]

    direction_states = {}
    for specification in directions:
        key = specification["key"]
        direction_states[key] = {
            "label": specification["label"],
            "color": specification["color"],
            "segments": [],
            "lines": np.empty((0, 3), dtype=float),
            "vanishing_point": None,
        }

    first_key = directions[0]["key"]
    state = {
        "active": first_key,
        "pending": [],
        "directions": direction_states,
    }

    fig, ax = plt.subplots(figsize=figsize)
    direction_toggle = ToggleButtons(
        options=[(item["label"], item["key"]) for item in directions],
        value=first_key,
        description="Selecting:",
        layout=Layout(width="auto"),
    )
    undo_button = Button(description="Undo active line")
    clear_active_button = Button(description="Clear active")
    clear_all_button = Button(description="Clear all")

    def update_estimates():
        for direction in direction_states.values():
            if direction["segments"]:
                direction["lines"] = np.array(
                    [
                        line_from_points(segment[0], segment[1])
                        for segment in direction["segments"]
                    ]
                )
            else:
                direction["lines"] = np.empty((0, 3), dtype=float)

            if len(direction["segments"]) >= 2:
                direction["vanishing_point"] = estimate_vanishing_point(
                    direction["lines"]
                )
            else:
                direction["vanishing_point"] = None

    def draw_line_across_image(line, color, active, alpha=None, linewidth=None):
        a, b, c = line
        candidates = []
        if abs(b) > 1e-12:
            for x in (0, width - 1):
                y = -(a * x + c) / b
                if 0 <= y <= height - 1:
                    candidates.append((x, y))
        if abs(a) > 1e-12:
            for y in (0, height - 1):
                x = -(b * y + c) / a
                if 0 <= x <= width - 1:
                    candidates.append((x, y))
        if len(candidates) >= 2:
            p1, p2 = candidates[:2]
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=color,
                alpha=alpha if alpha is not None else (0.45 if active else 0.22),
                linewidth=(
                    linewidth if linewidth is not None else (1.5 if active else 1)
                ),
            )

    def draw_off_image_vp_arrow(vp, color):
        center = np.array([(width - 1) / 2, (height - 1) / 2], dtype=float)
        direction = np.asarray(vp, dtype=float) - center
        if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) < 1e-12:
            return

        boundary_scales = []
        if direction[0] > 0:
            boundary_scales.append(((width - 1) - center[0]) / direction[0])
        elif direction[0] < 0:
            boundary_scales.append((0 - center[0]) / direction[0])
        if direction[1] > 0:
            boundary_scales.append(((height - 1) - center[1]) / direction[1])
        elif direction[1] < 0:
            boundary_scales.append((0 - center[1]) / direction[1])

        positive_scales = [scale for scale in boundary_scales if scale > 0]
        if not positive_scales:
            return
        edge_scale = min(positive_scales)
        edge = center + edge_scale * direction
        tail = center + 0.82 * edge_scale * direction
        ax.annotate(
            "",
            xy=edge,
            xytext=tail,
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.5},
        )

    def vp_summary(direction):
        vp = direction["vanishing_point"]
        count = len(direction["segments"])
        if vp is None:
            return f"{count} line(s); need at least 2"
        return f"{count} line(s); VP=({vp[0]:.1f}, {vp[1]:.1f})"

    def redraw():
        update_estimates()
        ax.clear()
        active_direction = direction_states[state["active"]]
        show(image, f"Active: {active_direction['label']}", ax)

        for key, direction in direction_states.items():
            is_active = key == state["active"]
            for segment, line in zip(direction["segments"], direction["lines"]):
                ax.plot(
                    segment[:, 0],
                    segment[:, 1],
                    color=direction["color"],
                    linewidth=3 if is_active else 2,
                    alpha=1 if is_active else 0.7,
                )
                ax.plot(
                    segment[:, 0],
                    segment[:, 1],
                    "o",
                    color=direction["color"],
                    markersize=4,
                    alpha=1 if is_active else 0.7,
                )
                draw_line_across_image(line, direction["color"], is_active)

            vp = direction["vanishing_point"]
            if vp is not None:
                inside = 0 <= vp[0] < width and 0 <= vp[1] < height
                if inside:
                    ax.plot(
                        vp[0],
                        vp[1],
                        "x",
                        color=direction["color"],
                        markersize=14,
                        markeredgewidth=3,
                    )
                else:
                    draw_off_image_vp_arrow(vp, direction["color"])

        vp_a = direction_states.get("facade_a", {}).get("vanishing_point")
        vp_b = direction_states.get("facade_b", {}).get("vanishing_point")
        if vp_a is not None and vp_b is not None:
            state["horizon"] = estimate_horizon(vp_a, vp_b)
            draw_line_across_image(
                state["horizon"],
                "red",
                active=True,
                alpha=0.9,
                linewidth=2.5,
            )
        else:
            state["horizon"] = None

        if state["pending"]:
            pending = np.asarray(state["pending"], dtype=float)
            ax.plot(
                pending[:, 0],
                pending[:, 1],
                "o",
                color=active_direction["color"],
                markersize=6,
            )

        summaries = [
            f"{direction['label']}: {vp_summary(direction)}"
            for direction in direction_states.values()
        ]
        if state["horizon"] is not None:
            summaries.append("Red line: horizon from the two facade vanishing points")
        summaries.append(f"Next click: endpoint {len(state['pending']) + 1}")
        ax.text(
            0.02,
            0.98,
            "\n".join(summaries),
            transform=ax.transAxes,
            color="white",
            va="top",
            bbox={"facecolor": "black", "alpha": 0.72, "pad": 4},
        )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        state["pending"].append([event.xdata, event.ydata])
        if len(state["pending"]) == 2:
            direction_states[state["active"]]["segments"].append(
                np.asarray(state["pending"], dtype=float)
            )
            state["pending"] = []
        redraw()

    def select_direction(change):
        if change["name"] != "value" or change["new"] == state["active"]:
            return
        state["active"] = change["new"]
        state["pending"] = []
        redraw()

    def undo_active_line(_):
        state["pending"] = []
        segments = direction_states[state["active"]]["segments"]
        if segments:
            segments.pop()
        redraw()

    def clear_active(_):
        state["pending"] = []
        direction_states[state["active"]]["segments"] = []
        redraw()

    def clear_all(_):
        state["pending"] = []
        for direction in direction_states.values():
            direction["segments"] = []
        redraw()

    direction_toggle.observe(select_direction, names="value")
    undo_button.on_click(undo_active_line)
    clear_active_button.on_click(clear_active)
    clear_all_button.on_click(clear_all)
    connection_id = fig.canvas.mpl_connect("button_press_event", on_click)
    display(
        VBox(
            [
                direction_toggle,
                HBox([undo_button, clear_active_button, clear_all_button]),
            ]
        )
    )
    redraw()

    state["figure"] = fig
    state["axis"] = ax
    state["direction_toggle"] = direction_toggle
    state["buttons"] = {
        "undo": undo_button,
        "clear_active": clear_active_button,
        "clear_all": clear_all_button,
    }
    state["connection_id"] = connection_id
    return state
