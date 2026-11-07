"""Reusable continuous representation, edge, color, and tone helpers for CSC320 Notebook 3."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from IPython.display import clear_output, display
from ipywidgets import Button, Checkbox, Dropdown, FloatSlider, HBox, IntSlider, Layout, Output, VBox, interactive_output
from image_utils import as_float_image, convolve_image, resize_image, rgb_to_gray, show_image as _show, to_uint8


def crop_center(image, size=180):
    """Return a centered square crop."""
    img = as_float_image(image)
    h, w = img.shape[:2]
    size = min(int(size), h, w)
    y0 = h // 2 - size // 2
    x0 = w // 2 - size // 2
    return img[y0 : y0 + size, x0 : x0 + size]


def make_gaussian_filter(size, sigma):
    """Create a normalized 2D Gaussian filter."""
    size = int(size) + (int(size) % 2 == 0)
    radius = size // 2
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    sigma = max(float(sigma), 1e-6)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(image, sigma=1.0):
    """Blur an image with a Gaussian filter."""
    if sigma <= 0:
        return as_float_image(image)
    size = int(2 * np.ceil(3 * sigma) + 1)
    return convolve_image(image, make_gaussian_filter(size, sigma))


def image_gradients(image, sigma=1.0):
    """Compute smoothed finite-difference image gradients.

    Parameters
    ----------
    image : array-like
        Input image.
    sigma : float
        Gaussian smoothing before derivatives.

    Returns
    -------
    gx, gy, magnitude : tuple[ndarray, ndarray, ndarray]
        Horizontal derivative, vertical derivative, and gradient magnitude.
    """
    gray = gaussian_blur(rgb_to_gray(image), sigma=sigma)
    dx = np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], dtype=float)
    dy = dx.T
    gx = -convolve_image(gray, dx)
    gy = -convolve_image(gray, dy)
    magnitude = np.hypot(gx, gy)
    return gx, gy, magnitude


def bilinear_sample(image, xs, ys):
    """Sample an image at floating point coordinates."""
    img = as_float_image(image)
    h, w = img.shape[:2]
    xs = np.clip(np.asarray(xs), 0, w - 1)
    ys = np.clip(np.asarray(ys), 0, h - 1)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = xs - x0
    wy = ys - y0
    return (
        (1 - wx) * (1 - wy) * img[y0, x0]
        + wx * (1 - wy) * img[y0, x1]
        + (1 - wx) * wy * img[y1, x0]
        + wx * wy * img[y1, x1]
    )


def _small_image(image, max_side=420, grayscale=False):
    img = rgb_to_gray(image) if grayscale else as_float_image(image)
    h, w = img.shape[:2]
    scale = min(1, max_side / max(h, w))
    return resize_image(img, scale=scale) if scale < 1 else img


def signal_sampling_widget():
    """Explore how samples can match many continuous 1D signals."""
    controls = {
        "frequency": FloatSlider(min=0.5, max=18, step=0.1, value=2.0, description="freq", readout_format=".1f"),
        "phase": FloatSlider(min=-np.pi, max=np.pi, step=0.1, value=0.0, description="phase", readout_format=".1f"),
        "samples": IntSlider(min=2, max=50, step=1, value=12, description="samples"),
        "show_1": Checkbox(value=False, description="show"),
        "candidate_1": FloatSlider(min=0.5, max=80, step=0.1, value=10.0, description="freq 1", readout_format=".1f"),
        "show_2": Checkbox(value=False, description="show"),
        "candidate_2": FloatSlider(min=0.5, max=80, step=0.1, value=14.0, description="freq 2", readout_format=".1f"),
        "show_3": Checkbox(value=False, description="show"),
        "candidate_3": FloatSlider(min=0.5, max=80, step=0.1, value=22.0, description="freq 3", readout_format=".1f"),
    }

    def draw(frequency, phase, samples, show_1, candidate_1, show_2, candidate_2, show_3, candidate_3):
        x = np.linspace(0, 1, 1000)
        y = np.cos(2 * np.pi * frequency * x + phase)
        xs = np.arange(samples) / samples
        ys = np.cos(2 * np.pi * frequency * xs + phase)

        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.plot(x, y, color="tab:blue", linewidth=2.5, label="chosen signal")
        for show, candidate, color, label in [
            (show_1, candidate_1, "tab:orange", "candidate 1"),
            (show_2, candidate_2, "tab:green", "candidate 2"),
            (show_3, candidate_3, "tab:purple", "candidate 3"),
        ]:
            if not show:
                continue
            y_candidate = np.cos(2 * np.pi * candidate * x + phase)
            ys_candidate = np.cos(2 * np.pi * candidate * xs + phase)
            error = np.max(np.abs(ys_candidate - ys))
            ax.plot(x, y_candidate, color=color, linewidth=1.8, alpha=0.8, label=f"{label}: max error {error:.2f}")
        ax.scatter(xs, ys, color="tab:red", edgecolor="white", zorder=3, label="samples")
        ax.vlines(xs, 0, ys, color="tab:red", alpha=0.35)
        ax.axhline(0, color="0.75", linewidth=1)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title("Find other cosine frequencies that pass through the samples")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    out = interactive_output(draw, controls)
    left = VBox([controls["frequency"], controls["phase"], controls["samples"]])
    for key in ("show_1", "show_2", "show_3"):
        controls[key].indent = False
        controls[key].layout = Layout(width="86px")
    for key in ("candidate_1", "candidate_2", "candidate_3"):
        controls[key].layout = Layout(width="260px")
    right = VBox(
        [
            HBox([controls["show_1"], controls["candidate_1"]], layout=Layout(gap="6px")),
            HBox([controls["show_2"], controls["candidate_2"]], layout=Layout(gap="6px")),
            HBox([controls["show_3"], controls["candidate_3"]], layout=Layout(gap="6px")),
        ]
    )
    display(HBox([left, right], layout=Layout(gap="32px")), out)
    return controls


def _cubic_interp(xs, xp, yp):
    values = []
    for x in xs:
        i = np.searchsorted(xp, x) - 1
        i = np.clip(i, 0, len(xp) - 2)
        i0, i1, i2, i3 = np.clip([i - 1, i, i + 1, i + 2], 0, len(xp) - 1)
        t = (x - xp[i1]) / max(xp[i2] - xp[i1], 1e-6)
        p0, p1, p2, p3 = yp[[i0, i1, i2, i3]]
        value = 0.5 * (
            (2 * p1)
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
        )
        values.append(value)
    return np.asarray(values)


def interpolation_1d_widget():
    """Compare nearest, linear, and cubic interpolation in 1D."""
    controls = {
        "samples": IntSlider(min=5, max=25, step=1, value=9, description="samples"),
        "frequency": FloatSlider(min=0.5, max=5.5, step=0.1, value=2.1, description="frequency", readout_format=".1f"),
        "method": Dropdown(options=["nearest", "linear", "cubic"], value="linear", description="method"),
    }

    def draw(samples, frequency, method):
        x = np.linspace(0, 1, 1000)
        truth = 0.75 * np.sin(2 * np.pi * frequency * x) + 0.25 * np.cos(2 * np.pi * 2.7 * x)
        xp = np.linspace(0, 1, samples)
        yp = 0.75 * np.sin(2 * np.pi * frequency * xp) + 0.25 * np.cos(2 * np.pi * 2.7 * xp)
        if method == "nearest":
            idx = np.argmin(np.abs(x[:, None] - xp[None, :]), axis=1)
            recon = yp[idx]
        elif method == "linear":
            recon = np.interp(x, xp, yp)
        else:
            recon = _cubic_interp(x, xp, yp)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(x, truth, color="0.65", linewidth=2, label="hidden continuous signal")
        ax.plot(x, recon, color="tab:blue", linewidth=2, label=f"{method} interpolation")
        ax.scatter(xp, yp, color="tab:red", zorder=3, label="samples")
        ax.set_ylim(-1.35, 1.35)
        ax.set_title("1D interpolation")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def interpolation_2d_widget(image):
    """Compare 2D interpolation artifacts on image crops."""
    img = as_float_image(image)
    h, w = img.shape[:2]
    slider_layout = Layout(width="230px")
    controls = {
        "x": IntSlider(min=0, max=max(0, w - 1), step=5, value=w // 2, description="x", continuous_update=False, layout=slider_layout),
        "y": IntSlider(min=0, max=max(0, h - 1), step=5, value=h // 2, description="y", continuous_update=False, layout=slider_layout),
        "zoom": FloatSlider(min=1, max=100, step=1, value=6, description="zoom", readout_format=".0f", continuous_update=False, layout=slider_layout),
    }

    def draw(x, y, zoom):
        target_shape = (210, 315)
        aspect = target_shape[1] / target_shape[0]
        zoom_h = max(4, int(round(min(h, w / aspect) / zoom)))
        zoom_w = max(4, int(round(zoom_h * aspect)))
        zoom_w = min(zoom_w, w)
        zoom_h = min(zoom_h, h)
        zx0 = int(np.clip(x - zoom_w // 2, 0, w - zoom_w))
        zy0 = int(np.clip(y - zoom_h // 2, 0, h - zoom_h))
        zoom_crop = img[zy0 : zy0 + zoom_h, zx0 : zx0 + zoom_w]
        methods = ["nearest", "bilinear", "bicubic", "lanczos"]
        fig = plt.figure(figsize=(7.2, 6.8))
        grid = fig.add_gridspec(3, 2, height_ratios=[0.85, 1.0, 1.0])
        ax_image = fig.add_subplot(grid[0, :])
        method_axes = [fig.add_subplot(grid[1 + i // 2, i % 2]) for i in range(4)]

        _show(img, "Image", ax_image)
        ax_image.add_patch(
            Rectangle(
                (zx0, zy0),
                zoom_w,
                zoom_h,
                fill=False,
                edgecolor="tab:red",
                linewidth=2,
            )
        )
        for ax_method, method in zip(method_axes, methods):
            enlarged = resize_image(zoom_crop, output_shape=target_shape, method=method)
            _show(enlarged, method, ax_method)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.04, wspace=0.04, hspace=0.20)
        plt.show()

    out = interactive_output(draw, controls)
    control_box = HBox(list(controls.values()), layout=Layout(width="700px", gap="6px"))
    display(VBox([control_box, out]))
    return controls


def _slice_coordinates(h, w, angle, offset):
    theta = np.deg2rad(angle)
    direction = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-direction[1], direction[0]])
    image_center = np.array([(w - 1) / 2, (h - 1) / 2])
    corners = np.array(
        [
            [0, 0],
            [w - 1, 0],
            [0, h - 1],
            [w - 1, h - 1],
        ]
    )
    max_offset = np.max(np.abs((corners - image_center) @ normal))
    for offset_scale in (1.0, 0.995, 0.98):
        center = image_center + offset * offset_scale * max_offset * normal
        intersections = []
        if abs(direction[0]) > 1e-9:
            for x_edge in (0, w - 1):
                alpha = (x_edge - center[0]) / direction[0]
                y_edge = center[1] + alpha * direction[1]
                if -1e-6 <= y_edge <= h - 1 + 1e-6:
                    intersections.append(alpha)
        if abs(direction[1]) > 1e-9:
            for y_edge in (0, h - 1):
                alpha = (y_edge - center[1]) / direction[1]
                x_edge = center[0] + alpha * direction[0]
                if -1e-6 <= x_edge <= w - 1 + 1e-6:
                    intersections.append(alpha)
        if len(intersections) >= 2 and max(intersections) - min(intersections) > 1e-6:
            break
    t = np.linspace(min(intersections), max(intersections), 260)
    xs = center[0] + t * direction[0]
    ys = center[1] + t * direction[1]
    return t, xs, ys


def _annotate_max(ax, t, values, label):
    idx = int(np.argmax(values))
    ax.scatter([t[idx]], [values[idx]], color="tab:red", s=35, zorder=4)
    ax.annotate(
        f"max {label}={values[idx]:.3f}",
        xy=(t[idx], values[idx]),
        xytext=(8, 8),
        textcoords="offset points",
        color="tab:red",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="tab:red", alpha=0.85),
    )


def _annotate_min(ax, t, values, label):
    idx = int(np.argmin(values))
    ax.scatter([t[idx]], [values[idx]], color="tab:green", s=35, zorder=4)
    ax.annotate(
        f"min {label}={values[idx]:.3f}",
        xy=(t[idx], values[idx]),
        xytext=(8, -14),
        textcoords="offset points",
        color="tab:green",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="tab:green", alpha=0.85),
    )


def derivative_clean_widget(image):
    """Explore image slices and derivatives without added noise."""
    base = _small_image(image, max_side=420, grayscale=True)
    h, w = base.shape
    controls = {
        "angle": FloatSlider(min=-180, max=180, step=1, value=0, description="slice angle", readout_format=".0f"),
        "offset": FloatSlider(min=-1, max=1, step=0.01, value=0.0, description="offset", readout_format=".2f"),
    }

    def draw(angle, offset):
        _, _, mag = image_gradients(base, sigma=0)
        t, xs, ys = _slice_coordinates(h, w, angle, offset)
        profile = bilinear_sample(base, xs, ys)
        derivative = np.gradient(profile, t)

        fig = plt.figure(figsize=(10.5, 7.0))
        grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25])
        ax_image = fig.add_subplot(grid[0, 0])
        ax_mag = fig.add_subplot(grid[1, 0])
        ax_profile = fig.add_subplot(grid[0, 1])
        ax_derivative = fig.add_subplot(grid[1, 1], sharex=ax_profile)
        _show(base, "Image", ax_image, cmap="gray", vmin=0, vmax=1)
        ax_image.plot(xs, ys, color="tab:red", linewidth=2)
        _show(mag, "Gradient magnitude", ax_mag, cmap="magma")
        ax_mag.plot(xs, ys, color="white", linewidth=2)
        ax_profile.plot(t, profile, color="tab:blue", label="intensity")
        _annotate_max(ax_profile, t, profile, "I")
        _annotate_min(ax_profile, t, profile, "I")
        ax_profile.set_title("Intensity along slice")
        ax_profile.set_ylabel("intensity")
        ax_derivative.plot(t, derivative, color="tab:purple")
        ax_derivative.axhline(0, color="0.75", linewidth=1)
        _annotate_max(ax_derivative, t, derivative, "dI/dt")
        _annotate_min(ax_derivative, t, derivative, "dI/dt")
        ax_derivative.set_title("Derivative along slice")
        ax_derivative.set_xlabel("position along slice")
        ax_derivative.set_ylabel("slope")
        plt.tight_layout()
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def derivative_gradient_reveal_widget(image):
    """Guess the strongest positive gradient location and direction, then reveal the answer."""
    base = _small_image(image, max_side=420, grayscale=True)
    h, w = base.shape
    gx, gy, mag = image_gradients(base, sigma=0)
    true_y, true_x = np.unravel_index(int(np.argmax(mag)), mag.shape)
    true_gradient = np.array([gx[true_y, true_x], gy[true_y, true_x]])
    true_norm = np.linalg.norm(true_gradient)
    true_direction = true_gradient / true_norm if true_norm > 0 else np.array([0.0, 0.0])
    true_angle = np.rad2deg(np.arctan2(true_direction[1], true_direction[0]))
    revealed = {"value": False}
    slider_layout = Layout(width="210px")

    controls = {
        "x": IntSlider(min=0, max=w - 1, step=1, value=w // 2, description="x", layout=slider_layout),
        "y": IntSlider(min=0, max=h - 1, step=1, value=h // 2, description="y", layout=slider_layout),
        "angle": FloatSlider(min=-180, max=180, step=1, value=0, description="angle", readout_format=".0f", layout=slider_layout),
    }
    reveal_button = Button(description="reveal truth", button_style="")
    out = Output()

    def draw(*_):
        theta = np.deg2rad(controls["angle"].value)
        guess_direction = np.array([np.cos(theta), np.sin(theta)])
        x = controls["x"].value
        y = controls["y"].value
        with out:
            clear_output(wait=True)
            fig, ax = plt.subplots(1, 2, figsize=(7.8, 4.4))
            _show(base, "Your guess", ax[0], cmap="gray", vmin=0, vmax=1)
            _show(mag, "Gradient magnitude", ax[1], cmap="magma")
            for axis in ax:
                axis.scatter([x], [y], color="tab:red", edgecolor="black", s=45, zorder=4)
                axis.arrow(
                    x,
                    y,
                    35 * guess_direction[0],
                    35 * guess_direction[1],
                    color="tab:red",
                    width=1.5,
                    head_width=8,
                    length_includes_head=True,
                    zorder=4,
                )
                if revealed["value"]:
                    axis.scatter([true_x], [true_y], color="cyan", edgecolor="black", s=45, zorder=5)
                    axis.arrow(
                        true_x,
                        true_y,
                        35 * true_direction[0],
                        35 * true_direction[1],
                        color="cyan",
                        width=1.5,
                        head_width=8,
                        length_includes_head=True,
                        zorder=5,
                    )
            if revealed["value"]:
                ax[1].set_title(f"Gradient magnitude | positive-gradient direction {true_angle:.0f} deg")
            fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02, wspace=0.08)
            plt.show()

    def reveal(_):
        revealed["value"] = True
        draw()

    for control in controls.values():
        control.observe(draw, names="value")
    reveal_button.on_click(reveal)
    draw()
    display(VBox([HBox(list(controls.values()) + [reveal_button], layout=Layout(gap="8px")), out]))
    return controls


def derivative_noise_widget(image):
    """Compare derivatives before and after adding noise and smoothing."""
    base = _small_image(image, max_side=420, grayscale=True)
    h, w = base.shape
    rng = np.random.default_rng(2)
    noise_field = rng.normal(size=base.shape)
    controls = {
        "noise": FloatSlider(min=0, max=0.35, step=0.01, value=0.08, description="noise", readout_format=".2f"),
        "sigma": FloatSlider(min=0, max=5, step=0.1, value=1.0, description="smooth", readout_format=".1f"),
        "angle": FloatSlider(min=-180, max=180, step=1, value=0, description="slice angle", readout_format=".0f"),
        "offset": FloatSlider(min=-1, max=1, step=0.01, value=0.0, description="offset", readout_format=".2f"),
    }

    def draw(noise, sigma, angle, offset):
        noisy = np.clip(base + noise * noise_field, 0, 1)
        smooth = gaussian_blur(noisy, sigma=sigma)
        t, xs, ys = _slice_coordinates(h, w, angle, offset)
        images = [base, noisy, smooth]
        labels = ["original", "noisy", "blurred"]
        colors = ["tab:blue", "tab:orange", "tab:green"]
        profiles = [bilinear_sample(frame, xs, ys) for frame in images]
        derivatives = [np.gradient(profile, t) for profile in profiles]

        fig = plt.figure(figsize=(9.2, 6.0))
        grid = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0])
        image_axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]), fig.add_subplot(grid[0, 4:6])]
        ax_profile = fig.add_subplot(grid[1, 0:3])
        ax_derivative = fig.add_subplot(grid[1, 3:6], sharex=ax_profile)
        for i, (frame, label) in enumerate(zip(images, labels)):
            _show(frame, label, image_axes[i], cmap="gray", vmin=0, vmax=1)
            image_axes[i].plot(xs, ys, color="tab:red", linewidth=2)
        for profile, label, color in zip(profiles, labels, colors):
            ax_profile.plot(t, profile, color=color, label=label)
        ax_profile.set_title("Intensity profiles")
        ax_profile.set_xlabel("position along slice")
        ax_profile.set_ylabel("intensity")
        for derivative, label, color in zip(derivatives, labels, colors):
            ax_derivative.plot(t, derivative, color=color, label=label)
        ax_derivative.axhline(0, color="0.75", linewidth=1)
        ax_derivative.set_title("Derivative profiles")
        ax_derivative.set_xlabel("position along slice")
        ax_derivative.set_ylabel("slope")
        ax_profile.legend(loc="upper right")
        ax_derivative.legend(loc="upper right")
        fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.08, wspace=0.06, hspace=0.26)
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def derivative_slice_widget(image):
    """Backward-compatible alias for the noise and smoothing widget."""
    return derivative_noise_widget(image)


def directional_derivative_widget(image):
    """Rotate a directional derivative and inspect response at a clicked point."""
    img = _small_image(image, max_side=360, grayscale=True)
    h, w = img.shape
    slider_layout = Layout(width="240px")
    controls = {
        "sigma": FloatSlider(min=0, max=4, step=0.1, value=1.0, description="smooth", readout_format=".1f", layout=slider_layout),
        "angle": FloatSlider(min=-180, max=180, step=1, value=0, description="direction", readout_format=".0f", layout=slider_layout),
    }
    state = {"x": w // 2, "y": h // 2}
    fig = plt.figure(figsize=(8.2, 3.6))
    ax_image = fig.add_axes([0.02, 0.15, 0.25, 0.74])
    ax_response = fig.add_axes([0.31, 0.15, 0.25, 0.74])
    cax = fig.add_axes([0.335, 0.07, 0.2, 0.035])
    ax_polar = fig.add_axes([0.66, 0.17, 0.29, 0.72], projection="polar")

    def render():
        x = state["x"]
        y = state["y"]
        sigma = controls["sigma"].value
        angle = controls["angle"].value
        gx, gy, mag = image_gradients(img, sigma=sigma)
        theta = np.deg2rad(angle)
        response = gx * np.cos(theta) + gy * np.sin(theta)
        point_gradient_angle = np.rad2deg(np.arctan2(gy[y, x], gx[y, x]))
        gradient_theta = np.deg2rad(point_gradient_angle)
        angles = np.linspace(-np.pi, np.pi, 361)
        values = gx[y, x] * np.cos(angles) + gy[y, x] * np.sin(angles)
        scale = max(np.max(np.abs(values)), 1e-9)
        chosen_value = response[y, x]
        chosen_radius = abs(chosen_value) / scale

        ax_image.clear()
        ax_response.clear()
        ax_polar.clear()
        cax.clear()

        _show(img, "Click a point", ax_image, cmap="gray", vmin=0, vmax=1)
        ax_image.scatter([x], [y], color="tab:red", s=35)
        ax_image.arrow(x, y, 28 * np.cos(theta), 28 * np.sin(theta), color="tab:red", width=1.2)
        ax_image.arrow(x, y, 28 * np.cos(gradient_theta), 28 * np.sin(gradient_theta), color="tab:green", width=1.0)

        limit = max(float(np.max(np.abs(response))), 1e-9)
        im = ax_response.imshow(response, cmap="coolwarm", vmin=-limit, vmax=limit)
        ax_response.set_title("Directional derivative map")
        ax_response.set_axis_off()
        ax_response.scatter([x], [y], color="black", s=25)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("negative to positive", labelpad=1)

        signed_radius = np.abs(values) / scale
        ax_polar.scatter(angles, signed_radius, c=values, cmap="coolwarm", vmin=-scale, vmax=scale, s=8)
        ax_polar.plot([theta, theta], [0, chosen_radius], color="tab:red", linewidth=2)
        ax_polar.scatter([theta], [chosen_radius], color="tab:red", s=30, label="directional derivative")
        ax_polar.plot([gradient_theta, gradient_theta], [0, 1], color="tab:green", linestyle="--", linewidth=2)
        ax_polar.scatter([gradient_theta], [1], color="tab:green", s=30, label="gradient direction")
        ax_polar.set_theta_direction(-1)
        ax_polar.set_theta_zero_location("E")
        ax_polar.set_rticks([])
        ax_polar.set_rlim(0, 1.05)
        ax_polar.set_title("Response by direction", pad=10)
        ax_polar.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=1, frameon=False)
        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes is ax_image and event.xdata is not None and event.ydata is not None:
            state["x"] = int(np.clip(round(event.xdata), 0, w - 1))
            state["y"] = int(np.clip(round(event.ydata), 0, h - 1))
            render()

    fig.canvas.mpl_connect("button_press_event", onclick)
    for control in controls.values():
        control.observe(lambda change: render(), names="value")

    render()
    display(VBox(list(controls.values())))
    return controls

def non_maximum_suppression(magnitude, gx, gy):
    """Keep gradient magnitudes that are locally largest along the gradient direction.

    Parameters
    ----------
    magnitude : ndarray, shape (H, W)
        Gradient magnitude image.
    gx, gy : ndarray, shape (H, W)
        Horizontal and vertical image derivatives.

    Returns
    -------
    thinned : ndarray, shape (H, W)
        Magnitude image after non-maximum suppression.
    """
    magnitude = np.asarray(magnitude, dtype=float)
    norm = np.hypot(gx, gy)
    ux = np.divide(gx, norm, out=np.zeros_like(magnitude), where=norm > 1e-9)
    uy = np.divide(gy, norm, out=np.zeros_like(magnitude), where=norm > 1e-9)
    y, x = np.indices(magnitude.shape)
    forward = bilinear_sample(magnitude, x + ux, y + uy)
    backward = bilinear_sample(magnitude, x - ux, y - uy)
    keep = (magnitude >= forward) & (magnitude >= backward)
    return magnitude * keep


def simple_canny_edges(image, sigma=1.5, low=0.05, high=0.15):
    """Compute a compact Canny-like edge map with suppression and hysteresis.

    Parameters
    ----------
    image : array-like
        Input image.
    sigma : float
        Smoothing before derivatives.
    low : float
        Low hysteresis threshold as a fraction of max gradient.
    high : float
        High hysteresis threshold as a fraction of max gradient.

    Returns
    -------
    edges, magnitude, thinned : tuple[ndarray, ndarray, ndarray]
        Binary edge map, gradient magnitude, and thinned magnitude.
    """
    gx, gy, mag = image_gradients(image, sigma=sigma)
    thinned = non_maximum_suppression(mag, gx, gy)
    scale = max(float(thinned.max()), 1e-6)
    strong = thinned >= high * scale
    weak = thinned >= low * scale
    edges = strong.copy()
    for _ in range(8):
        grown = edges.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= np.roll(np.roll(edges, dy, axis=0), dx, axis=1)
        edges = weak & grown
    return edges.astype(float), mag, thinned


def canny_widget(image):
    """Explore a Canny-like edge detector as a parameterized system."""
    img = _small_image(image, max_side=420, grayscale=True)
    h, w = img.shape
    state = {"x": w // 2, "y": h // 2}
    slider_layout = Layout(width="260px")
    controls = {
        "sigma": FloatSlider(min=0, max=5, step=0.1, value=1.5, description="smooth", readout_format=".1f", layout=slider_layout),
        "low": FloatSlider(min=0, max=0.5, step=0.01, value=0.06, description="low", readout_format=".2f", layout=slider_layout),
        "high": FloatSlider(min=0, max=0.8, step=0.01, value=0.18, description="high", readout_format=".2f", layout=slider_layout),
    }
    move_buttons = {
        "left": Button(description="left", layout=Layout(width="62px")),
        "right": Button(description="right", layout=Layout(width="62px")),
        "up": Button(description="up", layout=Layout(width="62px")),
        "down": Button(description="down", layout=Layout(width="62px")),
    }

    fig = plt.figure(figsize=(8.0, 4.8))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1.0, 1.0])
    ax_input = fig.add_subplot(grid[0, 0])
    ax_mag = fig.add_subplot(grid[0, 1])
    ax_thin = fig.add_subplot(grid[0, 2])
    ax_edges = fig.add_subplot(grid[1, 0])
    ax_blank = fig.add_subplot(grid[1, 1])
    ax_profile = fig.add_subplot(grid[1, 2])
    ax_blank.set_axis_off()
    fig.subplots_adjust(left=0.04, right=0.98, top=0.91, bottom=0.13, wspace=0.22, hspace=0.34)

    def render():
        sigma = controls["sigma"].value
        low = controls["low"].value
        high = max(controls["high"].value, low)
        x = state["x"]
        y = state["y"]
        gx, gy, mag = image_gradients(img, sigma=sigma)
        thinned = non_maximum_suppression(mag, gx, gy)
        scale = max(float(thinned.max()), 1e-6)
        strong = thinned >= high * scale
        weak = thinned >= low * scale
        edges = strong.copy()
        for _ in range(8):
            grown = edges.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    grown |= np.roll(np.roll(edges, dy, axis=0), dx, axis=1)
            edges = weak & grown

        direction_norm = max(float(np.hypot(gx[y, x], gy[y, x])), 1e-9)
        ux = gx[y, x] / direction_norm
        uy = gy[y, x] / direction_norm
        t = np.linspace(-4, 4, 181)
        xs = x + t * ux
        ys = y + t * uy
        profile = bilinear_sample(mag, xs, ys)
        comparison_t = np.array([-1.0, 0.0, 1.0])
        comparison = bilinear_sample(mag, x + comparison_t * ux, y + comparison_t * uy)
        survives = comparison[1] >= comparison[0] and comparison[1] >= comparison[2]

        for ax in (ax_input, ax_mag, ax_thin, ax_edges, ax_profile, ax_blank):
            ax.clear()
        _show(img, "Input", ax_input, cmap="gray", vmin=0, vmax=1)
        _show(mag, "Gradient magnitude (click for direction)", ax_mag, cmap="magma")
        _show(thinned, "After NMS", ax_thin, cmap="magma")
        _show(edges.astype(float), "Final edges", ax_edges, cmap="gray", vmin=0, vmax=1)
        ax_blank.set_axis_off()

        arrow_length = 22
        for ax in (ax_mag, ax_thin):
            ax.scatter([x], [y], color="cyan", edgecolor="black", s=35, zorder=3)
            ax.arrow(x, y, arrow_length * ux, arrow_length * uy, color="cyan", width=0.7, head_width=5, zorder=3)
            ax.plot([x - arrow_length * ux, x + arrow_length * ux], [y - arrow_length * uy, y + arrow_length * uy], color="cyan", linewidth=1.2)

        center_color = "tab:green" if survives else "tab:red"
        ax_profile.plot(t, profile, color="black", linewidth=2)
        ax_profile.scatter(comparison_t, comparison, color=["tab:orange", center_color, "tab:orange"], s=[45, 65, 45], zorder=3)
        ax_profile.axvline(-1, color="tab:orange", linestyle=":", linewidth=1.5)
        ax_profile.axvline(0, color=center_color, linestyle="-", linewidth=1.5)
        ax_profile.axvline(1, color="tab:orange", linestyle=":", linewidth=1.5)
        ax_profile.set_title("1D NMS profile")
        ax_profile.set_xlabel("position")
        ax_profile.set_ylabel("gradient magnitude")
        ax_profile.text(
            0.02,
            0.92,
            "survives NMS" if survives else "suppressed by NMS",
            color=center_color,
            transform=ax_profile.transAxes,
            fontsize=10,
            weight="bold",
        )
        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes is ax_mag and event.xdata is not None and event.ydata is not None:
            state["x"] = int(np.clip(round(event.xdata), 0, w - 1))
            state["y"] = int(np.clip(round(event.ydata), 0, h - 1))
            render()

    def move(dx, dy):
        state["x"] = int(np.clip(state["x"] + dx, 0, w - 1))
        state["y"] = int(np.clip(state["y"] + dy, 0, h - 1))
        render()

    def onkey(event):
        steps = {
            "left": (-1, 0),
            "right": (1, 0),
            "up": (0, -1),
            "down": (0, 1),
        }
        if event.key in steps:
            move(*steps[event.key])

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    for control in controls.values():
        control.observe(lambda change: render(), names="value")
    move_buttons["left"].on_click(lambda _: move(-1, 0))
    move_buttons["right"].on_click(lambda _: move(1, 0))
    move_buttons["up"].on_click(lambda _: move(0, -1))
    move_buttons["down"].on_click(lambda _: move(0, 1))

    render()
    display(VBox([*controls.values(), HBox([move_buttons["left"], move_buttons["right"], move_buttons["up"], move_buttons["down"]])]))
    return controls


def rgb_to_hsv(image):
    """Convert RGB image to HSV with channels in [0, 1]."""
    rgb = as_float_image(image)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    mask = diff > 1e-6
    h[(mx == r) & mask] = ((g - b) / diff % 6)[(mx == r) & mask]
    h[(mx == g) & mask] = ((b - r) / diff + 2)[(mx == g) & mask]
    h[(mx == b) & mask] = ((r - g) / diff + 4)[(mx == b) & mask]
    h /= 6
    s = np.where(mx > 1e-6, diff / mx, 0)
    v = mx
    return np.stack([h, s, v], axis=-1)


def rgb_to_lab(image):
    """Convert sRGB to CIE Lab, scaled to roughly [0, 1] for display."""
    rgb = as_float_image(image)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
    )
    xyz = linear @ M.T
    white = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / white
    eps = 216 / 24389
    kappa = 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L / 100, (a + 128) / 255, (b + 128) / 255], axis=-1)


def color_space_channels(image, space="RGB"):
    """Return a three-channel representation and channel names."""
    img = as_float_image(image)
    if space == "RGB":
        return img, ["R", "G", "B"]
    if space == "HSV":
        return rgb_to_hsv(img), ["H", "S", "V"]
    if space == "Lab":
        return rgb_to_lab(img), ["L", "a", "b"]
    y = rgb_to_gray(img)
    cb = 0.5 + (img[..., 2] - y) * 0.564
    cr = 0.5 + (img[..., 0] - y) * 0.713
    return np.stack([y, cb, cr], axis=-1), ["Y", "Cb", "Cr"]


def color_space_widget(image):
    """Inspect channels and threshold masks in different color spaces."""
    img = _small_image(image, max_side=520)
    controls = {
        "space": Dropdown(options=["RGB", "HSV", "Lab", "YCbCr"], value="RGB", description="space"),
        "channel": Dropdown(options=[0, 1, 2], value=0, description="channel"),
        "low": FloatSlider(min=0, max=1, step=0.01, value=0.25, description="low", readout_format=".2f"),
        "high": FloatSlider(min=0, max=1, step=0.01, value=0.75, description="high", readout_format=".2f"),
    }

    def draw(space, channel, low, high):
        rep, names = color_space_channels(img, space)
        ch = rep[..., channel]
        mask = (ch >= low) & (ch <= high)
        segmented = img.copy()
        segmented[~mask] *= 0.15
        with plt.ioff():
            fig, ax = plt.subplots(2, 2, figsize=(7.8, 6.0))
        ax = ax.ravel()
        _show(img, "Input", ax[0])
        _show(ch, f"{space} {names[channel]}", ax[1], cmap="gray", vmin=0, vmax=1)
        _show(mask.astype(float), "Mask", ax[2], cmap="gray", vmin=0, vmax=1)
        _show(segmented, "Masked input", ax[3])
        plt.tight_layout()
        display(fig)
        plt.close(fig)

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def srgb_to_xyz(image):
    """Convert sRGB values in [0, 1] to CIE XYZ using the D65 sRGB matrix."""
    rgb = as_float_image(image)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    return linear @ matrix.T


def xyz_to_xy(xyz):
    """Convert XYZ values to xy chromaticity coordinates."""
    xyz = np.asarray(xyz, dtype=float)
    total = xyz[..., 0] + xyz[..., 1] + xyz[..., 2]
    x = np.divide(xyz[..., 0], total, out=np.zeros_like(total), where=total > 1e-9)
    y = np.divide(xyz[..., 1], total, out=np.zeros_like(total), where=total > 1e-9)
    return x, y


def spectral_locus_xy():
    """Approximate the CIE 1931 spectral locus in xy coordinates."""
    wavelengths = np.arange(380, 701, 5)
    x_bar = (
        1.056 * np.exp(-0.5 * ((wavelengths - 599.8) / 37.9) ** 2)
        + 0.362 * np.exp(-0.5 * ((wavelengths - 442.0) / 16.0) ** 2)
        - 0.065 * np.exp(-0.5 * ((wavelengths - 501.1) / 20.4) ** 2)
    )
    y_bar = (
        0.821 * np.exp(-0.5 * ((wavelengths - 568.8) / 46.9) ** 2)
        + 0.286 * np.exp(-0.5 * ((wavelengths - 530.9) / 16.3) ** 2)
    )
    z_bar = (
        1.217 * np.exp(-0.5 * ((wavelengths - 437.0) / 11.8) ** 2)
        + 0.681 * np.exp(-0.5 * ((wavelengths - 459.0) / 26.0) ** 2)
    )
    return xyz_to_xy(np.stack([x_bar, y_bar, z_bar], axis=1))


def image_chromaticity_widget(image):
    """Plot image pixels in CIE xy chromaticity space."""
    img = _small_image(image, max_side=460)
    pixels = img.reshape(-1, 3)
    xyz = srgb_to_xyz(pixels)
    x, y = xyz_to_xy(xyz)
    brightness = xyz[:, 1]
    keep = brightness > np.percentile(brightness, 5)
    rng = np.random.default_rng(0)
    kept_indices = np.flatnonzero(keep)
    max_shown_pixels = max(500, int(0.9 * kept_indices.size))
    default_shown_pixels = min(5000, max_shown_pixels)
    controls = {
        "brightness": FloatSlider(min=0, max=1.0, step=0.01, value=0.03, description="min brightness", readout_format=".2f"),
        "sample": IntSlider(min=500, max=max_shown_pixels, step=500, value=default_shown_pixels, description="shown pixels"),
    }
    locus_x, locus_y = spectral_locus_xy()
    rgb_primaries = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    primary_x, primary_y = xyz_to_xy(srgb_to_xyz(rgb_primaries))
    white_x, white_y = xyz_to_xy(srgb_to_xyz(np.array([[1, 1, 1]], dtype=float)))

    def draw(brightness, sample):
        valid = xyz[:, 1] > brightness
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size > sample:
            valid_indices = rng.choice(valid_indices, size=sample, replace=False)
        with plt.ioff():
            fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.9), gridspec_kw={"width_ratios": [1.0, 1.15]})
        _show(img, "Input", ax[0])
        ax[1].plot(locus_x, locus_y, color="black", linewidth=1.5)
        ax[1].plot(primary_x, primary_y, color="black", linewidth=1.2)
        ax[1].scatter(
            x[valid_indices],
            y[valid_indices],
            c=np.clip(pixels[valid_indices], 0, 1),
            s=7,
            alpha=0.35,
            linewidths=0,
        )
        ax[1].scatter(white_x, white_y, color="white", edgecolor="black", s=45, zorder=3)
        ax[1].set_title("Image pixels in xy chromaticity")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")
        ax[1].set_xlim(0, 0.8)
        ax[1].set_ylim(0, 0.9)
        ax[1].set_aspect("equal", adjustable="box")
        ax[1].grid(True, color="0.9", linewidth=0.8)
        plt.tight_layout()
        display(fig)
        plt.close(fig)

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def apply_color_cast(image, gains=(1.0, 1.0, 1.0)):
    """Apply per-channel gains and clip to [0, 1]."""
    return np.clip(as_float_image(image) * np.asarray(gains), 0, 1)


def simulate_underwater(image):
    """Create a simple underwater-like color cast."""
    img = gaussian_blur(as_float_image(image), sigma=0.8)
    return apply_color_cast(img, gains=(0.45, 0.85, 1.35))


def white_balance(image, method="gray world", patch=None):
    """Apply a simple white-balance correction.

    Parameters
    ----------
    image : array-like
        Input RGB image.
    method : {"gray world", "brightest", "patch"}
        White-balance method.
    patch : tuple[int, int, int] or None
        Patch as (x, y, radius) for method="patch".

    Returns
    -------
    corrected, gains : tuple[ndarray, ndarray]
        Corrected image and RGB gains.
    """
    img = as_float_image(image)
    if method == "brightest":
        gray = rgb_to_gray(img)
        threshold = np.percentile(gray, 99)
        estimate = img[gray >= threshold].mean(axis=0)
    elif method == "patch" and patch is not None:
        x, y, r = patch
        estimate = img[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1].mean(axis=(0, 1))
    else:
        estimate = img.mean(axis=(0, 1))
    gains = estimate.mean() / np.maximum(estimate, 1e-6)
    return np.clip(img * gains, 0, 1), gains


def white_balance_widget(images):
    """Compare simple white-balance assumptions on different images."""
    names = list(images)
    prepared = {name: _small_image(img, max_side=520) for name, img in images.items()}
    first = prepared[names[0]]
    h, w = first.shape[:2]
    controls = {
        "image_name": Dropdown(options=names, value=names[0], description="image"),
        "method": Dropdown(options=["gray world", "brightest", "patch"], value="gray world", description="method"),
        "x": IntSlider(min=0, max=w - 1, step=1, value=w // 2, description="x"),
        "y": IntSlider(min=0, max=h - 1, step=1, value=h // 2, description="y"),
        "radius": IntSlider(min=2, max=50, step=1, value=12, description="patch"),
    }

    def draw(image_name, method, x, y, radius):
        img = prepared[image_name]
        h, w = img.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        corrected, gains = white_balance(img, method=method, patch=(x, y, radius))
        with plt.ioff():
            fig, ax = plt.subplots(1, 3, figsize=(8.6, 3.0), gridspec_kw={"width_ratios": [1, 1, 1]})
        _show(img, f"Input: {image_name}", ax[0])
        if method == "patch":
            ax[0].add_patch(plt.Rectangle((x - radius, y - radius), 2 * radius, 2 * radius, fill=False, color="yellow", linewidth=2))
        _show(corrected, "Corrected", ax[1])
        ax[2].bar(["R", "G", "B"], gains, color=["tab:red", "tab:green", "tab:blue"])
        ax[2].set_box_aspect(1)
        ax[2].set_ylim(0, 3)
        ax[2].set_title("Channel gains")
        fig.subplots_adjust(left=0.04, right=0.98, top=0.86, bottom=0.12, wspace=0.18)
        display(fig)
        plt.close(fig)

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def gamma_encode(linear, gamma=2.2):
    """Encode linear intensities with a gamma curve."""
    return np.clip(as_float_image(linear), 0, 1) ** (1 / gamma)


def gamma_decode(encoded, gamma=2.2):
    """Decode gamma-encoded intensities to linear values."""
    return np.clip(as_float_image(encoded), 0, 1) ** gamma


def tone_map(image, exposure=0, contrast=1.0, gamma=2.2):
    """Apply exposure, contrast, and gamma display mapping."""
    linear = np.clip(as_float_image(image) * (2 ** exposure), 0, 1)
    contrasted = np.clip((linear - 0.5) * contrast + 0.5, 0, 1)
    return gamma_encode(contrasted, gamma=gamma)


def gamma_tone_widget(image):
    """Explore black point, white point, and gamma tone mapping."""
    img = _small_image(image, max_side=520)
    input_img = np.clip(0.18 + 0.62 * img, 0, 1)
    controls = {
        "black": FloatSlider(min=0, max=1, step=0.01, value=0.0, description="black", readout_format=".2f"),
        "white": FloatSlider(min=0, max=1, step=0.01, value=1.0, description="white", readout_format=".2f"),
        "gamma": FloatSlider(min=0.3, max=3.5, step=0.05, value=1.0, description="gamma", readout_format=".2f"),
    }

    def draw(black, white, gamma):
        white = max(white, black + 0.01)
        stretched = np.clip((input_img - black) / (white - black), 0, 1)
        mapped = stretched ** (1 / gamma)
        gray = rgb_to_gray(input_img)
        x = np.linspace(0, 1, 256)
        curve = np.clip((x - black) / (white - black), 0, 1) ** (1 / gamma)
        with plt.ioff():
            fig, ax = plt.subplots(2, 2, figsize=(8.2, 5.8), gridspec_kw={"height_ratios": [1.0, 0.85]})
        _show(input_img, "Input", ax[0, 0])
        _show(mapped, "Tone mapped", ax[0, 1])
        ax[1, 0].hist(gray.ravel(), bins=80, range=(0, 1), color="0.35")
        ax[1, 0].axvline(black, color="black", linewidth=2)
        ax[1, 0].axvline(white, color="tab:red", linewidth=2)
        ax[1, 0].set_title("Input histogram")
        ax[1, 0].set_xlabel("input value")
        ax[1, 0].set_ylabel("pixel count")
        ax[1, 0].set_xlim(0, 1)
        ax[1, 1].plot(x, curve, color="black", linewidth=2)
        ax[1, 1].plot([0, 1], [0, 1], color="0.75", linestyle="--")
        ax[1, 1].set_title("Tone curve")
        ax[1, 1].set_xlabel("input value")
        ax[1, 1].set_ylabel("output value")
        ax[1, 1].set_xlim(0, 1)
        ax[1, 1].set_ylim(0, 1)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.1, wspace=0.2, hspace=0.35)
        display(fig)
        plt.close(fig)

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls
