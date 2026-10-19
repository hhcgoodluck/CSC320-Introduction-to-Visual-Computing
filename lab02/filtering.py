"""Reusable filtering, Fourier, and sampling functions for CSC320 Notebook 2.

The implementations are intentionally compact and dependency-light so students
can inspect, reuse, and modify them in project milestones.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import Dropdown, FloatSlider, HBox, IntSlider, Layout, VBox, interactive_output
from image_utils import as_float_image, convolve_image, resize_image, rgb_to_gray, show_image as _show, to_uint8


HIDDEN_HYBRID_TARGET_PARAMS = {
    "low_sigma": 6.0,
    "high_sigma": 2.2,
    "high_weight": 1.6,
}


def translate_image(image, tx=0, ty=0, fill=0.0):
    """Translate an image without wrapping.

    Parameters
    ----------
    image : array-like
        Input grayscale or RGB image.
    tx : int
        Horizontal translation in pixels. Positive moves right.
    ty : int
        Vertical translation in pixels. Positive moves down.
    fill : float
        Value used for newly exposed pixels.

    Returns
    -------
    translated : ndarray
        Translated image with the same shape as the input.
    """
    img = as_float_image(image)
    out = np.full_like(img, fill, dtype=float)
    tx, ty = int(tx), int(ty)
    h, w = img.shape[:2]

    src_x0 = max(0, -tx)
    src_x1 = min(w, w - tx)
    dst_x0 = max(0, tx)
    dst_x1 = min(w, w + tx)
    src_y0 = max(0, -ty)
    src_y1 = min(h, h - ty)
    dst_y0 = max(0, ty)
    dst_y1 = min(h, h + ty)

    if src_x1 > src_x0 and src_y1 > src_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return out


def _odd(value):
    value = int(value)
    return value + 1 if value % 2 == 0 else value


def _as_pair(value):
    if np.isscalar(value):
        return value, value
    return value


def make_box_filter(size):
    """Create a normalized box filter.

    Parameters
    ----------
    size : int or tuple[int, int]
        Odd filter height and width. A single integer makes a square filter.

    Returns
    -------
    kernel : ndarray, shape (height, width)
        Box filter whose entries sum to 1.
    """
    size_y, size_x = _as_pair(size)
    size_y, size_x = _odd(size_y), _odd(size_x)
    return np.ones((size_y, size_x), dtype=float) / (size_y * size_x)


def make_gaussian_filter(size, sigma):
    """Create a normalized 2D Gaussian filter.

    Parameters
    ----------
    size : int or tuple[int, int]
        Odd filter height and width. A single integer makes a square filter.
    sigma : float or tuple[float, float]
        Gaussian standard deviation as (sigma_y, sigma_x). A single float
        makes an isotropic Gaussian.

    Returns
    -------
    kernel : ndarray, shape (height, width)
        Gaussian filter whose entries sum to 1.
    """
    size_y, size_x = _as_pair(size)
    sigma_y, sigma_x = _as_pair(sigma)
    size_y, size_x = _odd(size_y), _odd(size_x)
    radius_y, radius_x = size_y // 2, size_x // 2
    y, x = np.mgrid[-radius_y : radius_y + 1, -radius_x : radius_x + 1]
    sigma_y = max(float(sigma_y), 1e-6)
    sigma_x = max(float(sigma_x), 1e-6)
    kernel = np.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))
    return kernel / kernel.sum()


def make_derivative_filter(kind="sobel_x"):
    """Create a small derivative filter.

    Parameters
    ----------
    kind : {"dx", "dy", "sobel_x", "sobel_y"}
        Derivative filter type.

    Returns
    -------
    kernel : ndarray
        Derivative filter kernel.
    """
    filters = {
        "dx": np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], dtype=float),
        "dy": np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], dtype=float),
        "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float) / 8,
        "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float) / 8,
    }
    return filters[kind]


def make_sharpening_filter(amount=1.0):
    """Create a simple 3x3 sharpening filter.

    Parameters
    ----------
    amount : float
        Strength of the four-neighbor sharpening term.

    Returns
    -------
    kernel : ndarray, shape (3, 3)
        Sharpening kernel.
    """
    a = float(amount)
    return np.array([[0, -a, 0], [-a, 1 + 4 * a, -a], [0, -a, 0]], dtype=float)


def gaussian_blur(image, sigma=2.0, size=None):
    """Blur an image with a Gaussian filter.

    Parameters
    ----------
    image : array-like
        Input image.
    sigma : float
        Gaussian standard deviation in pixels.
    size : int or None
        Kernel size. If None, choose a size from sigma.

    Returns
    -------
    blurred : ndarray
        Gaussian-blurred image.
    """
    if size is None:
        size = int(2 * np.ceil(3 * sigma) + 1)
    return convolve_image(image, make_gaussian_filter(size, sigma))


def image_gradients(image, sigma=1.0, method="sobel"):
    """Compute image gradients.

    Parameters
    ----------
    image : array-like
        Input image.
    sigma : float
        Gaussian smoothing applied before derivatives.
    method : {"sobel", "finite"}
        Derivative filter family.

    Returns
    -------
    Ix, Iy, magnitude, orientation : tuple[ndarray, ndarray, ndarray, ndarray]
        Horizontal derivative, vertical derivative, gradient magnitude, and
        orientation in radians.
    """
    gray = rgb_to_gray(image)
    smoothed = gaussian_blur(gray, sigma=sigma) if sigma > 0 else gray
    x_filter = "sobel_x" if method == "sobel" else "dx"
    y_filter = "sobel_y" if method == "sobel" else "dy"
    Ix = convolve_image(smoothed, make_derivative_filter(x_filter))
    Iy = convolve_image(smoothed, make_derivative_filter(y_filter))
    magnitude = np.sqrt(Ix**2 + Iy**2)
    orientation = np.arctan2(Iy, Ix)
    return Ix, Iy, magnitude, orientation


def unsharp_mask(image, radius=2.0, amount=1.0):
    """Sharpen an image by adding high-frequency residuals.

    Parameters
    ----------
    image : array-like
        Input image.
    radius : float
        Gaussian blur radius used to define low frequencies.
    amount : float
        Strength of the high-frequency residual.

    Returns
    -------
    sharpened, blurred, detail : tuple[ndarray, ndarray, ndarray]
        Sharpened image, blurred image, and high-frequency detail image.
    """
    img = as_float_image(image)
    blurred = gaussian_blur(img, sigma=radius)
    detail = img - blurred
    return np.clip(img + amount * detail, 0, 1), blurred, detail


def bilateral_filter(image, spatial_sigma=3.0, range_sigma=0.15, radius=None):
    """Apply a simple bilateral filter.

    Parameters
    ----------
    image : array-like, shape (H, W) or (H, W, C)
        Input image. Use a small image or crop for interactive use.
    spatial_sigma : float
        Standard deviation for spatial distance.
    range_sigma : float
        Standard deviation for intensity or color distance.
    radius : int or None
        Neighborhood radius. If None, choose from spatial_sigma.

    Returns
    -------
    filtered : ndarray
        Edge-preserving smoothed image.
    """
    img = as_float_image(image)
    if radius is None:
        radius = max(1, int(np.ceil(2 * spatial_sigma)))
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    spatial = np.exp(-(x**2 + y**2) / (2 * spatial_sigma**2))
    pad_width = ((radius, radius), (radius, radius)) if img.ndim == 2 else ((radius, radius), (radius, radius), (0, 0))
    padded = np.pad(img, pad_width, mode="reflect")
    out = np.zeros_like(img, dtype=float)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            patch = padded[row : row + 2 * radius + 1, col : col + 2 * radius + 1]
            center = img[row, col]
            if img.ndim == 2:
                range_weight = np.exp(-((patch - center) ** 2) / (2 * range_sigma**2))
            else:
                color_distance2 = np.sum((patch - center) ** 2, axis=2)
                range_weight = np.exp(-color_distance2 / (2 * range_sigma**2))
            weights = spatial * range_weight
            if img.ndim == 2:
                out[row, col] = np.sum(weights * patch) / np.sum(weights)
            else:
                out[row, col] = np.sum(patch * weights[..., None], axis=(0, 1)) / np.sum(weights)
    return out


def diagonal_edge_image(size=140):
    """Create a synthetic grayscale image with a diagonal edge."""
    y, x = np.indices((size, size))
    return (x + y > size).astype(float)


def bilateral_weights_for_pixel(image, center, spatial_sigma=3.0, range_sigma=0.15, radius=8):
    """Compute bilateral-filter weights for one pixel.

    Parameters
    ----------
    image : array-like, shape (H, W)
        Grayscale input image.
    center : tuple[int, int]
        Center pixel as (row, col).
    spatial_sigma : float
        Standard deviation for spatial distance.
    range_sigma : float
        Standard deviation for intensity distance.
    radius : int
        Neighborhood radius.

    Returns
    -------
    patch, spatial_weights, range_weights, bilateral_weights : tuple[ndarray, ndarray, ndarray, ndarray]
        Local image patch and the three weight arrays for the center pixel.
    """
    img = rgb_to_gray(image)
    row, col = center
    radius = int(radius)
    padded = np.pad(img, radius, mode="reflect")
    patch = padded[row : row + 2 * radius + 1, col : col + 2 * radius + 1]
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    spatial_sigma = max(float(spatial_sigma), 1e-6)
    range_sigma = max(float(range_sigma), 1e-6)
    spatial_weights = np.exp(-(x**2 + y**2) / (2 * spatial_sigma**2))
    range_weights = np.exp(-((patch - img[row, col]) ** 2) / (2 * range_sigma**2))
    bilateral_weights = spatial_weights * range_weights
    bilateral_weights = bilateral_weights / bilateral_weights.sum()
    return patch, spatial_weights, range_weights, bilateral_weights


def bilateral_edge_widget(size=140):
    """Explore bilateral filtering on a noisy diagonal edge."""
    clean = diagonal_edge_image(size)
    state = {
        "center": (size // 2, size // 2),
        "clean": clean,
        "noisy": clean.copy(),
        "filtered": clean.copy(),
    }
    controls = {
        "noise": FloatSlider(min=0, max=0.5, step=0.01, value=0.12, description="noise", readout_format=".2f"),
        "sigma_s": FloatSlider(min=0.5, max=14, step=0.1, value=4.0, description="sigma_s", readout_format=".1f"),
        "sigma_r": FloatSlider(min=0.02, max=5.0, step=0.02, value=0.12, description="sigma_r", readout_format=".2f"),
    }
    fig, ax = plt.subplots(2, 3, figsize=(11, 7))
    ax = ax.ravel()

    def neighborhood_radius():
        return max(1, int(np.ceil(2 * controls["sigma_s"].value)))

    def current_images():
        rng = np.random.default_rng(0)
        noisy = np.clip(clean + controls["noise"].value * rng.normal(size=clean.shape), 0, 1)
        filtered = bilateral_filter(
            noisy,
            spatial_sigma=controls["sigma_s"].value,
            range_sigma=controls["sigma_r"].value,
            radius=neighborhood_radius(),
        )
        return noisy, filtered

    def redraw(*_):
        noisy, filtered = current_images()
        state["noisy"] = noisy
        state["filtered"] = filtered
        row, col = state["center"]
        patch, spatial_w, range_w, bilateral_w = bilateral_weights_for_pixel(
            noisy,
            (row, col),
            spatial_sigma=controls["sigma_s"].value,
            range_sigma=controls["sigma_r"].value,
            radius=neighborhood_radius(),
        )
        selected_values = patch * bilateral_w / bilateral_w.max()

        for axis in ax:
            axis.clear()
        _show(noisy, "Noisy diagonal edge", ax[0], cmap="gray", vmin=0, vmax=1)
        ax[0].plot(col, row, "ro", markersize=5)
        circle = plt.Circle((col, row), controls["sigma_s"].value, color="red", fill=False, linewidth=2)
        ax[0].add_patch(circle)
        _show(range_w, "range weights", ax[1], cmap="viridis", vmin=0, vmax=1)
        _show(selected_values, "selected values", ax[2], cmap="gray", vmin=0, vmax=1)
        ax[3].imshow(patch, cmap="gray", vmin=0, vmax=1)
        patch_radius = patch.shape[0] // 2
        patch_circle = plt.Circle(
            (patch_radius, patch_radius),
            controls["sigma_s"].value,
            color="red",
            fill=False,
            linewidth=2,
        )
        ax[3].add_patch(patch_circle)
        ax[3].set_title("spatial neighborhood")
        ax[3].set_axis_off()
        _show(filtered, "Bilateral result", ax[4], cmap="gray", vmin=0, vmax=1)
        _show(gaussian_blur(noisy, sigma=controls["sigma_s"].value), "Gaussian result", ax[5], cmap="gray", vmin=0, vmax=1)
        plt.tight_layout()
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax[0] or event.xdata is None or event.ydata is None:
            return
        state["center"] = (
            int(np.clip(round(event.ydata), 0, size - 1)),
            int(np.clip(round(event.xdata), 0, size - 1)),
        )
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)

    for control in controls.values():
        control.observe(redraw, names="value")
    display(VBox([controls["noise"], HBox([controls["sigma_s"], controls["sigma_r"]])]))
    redraw()
    state["controls"] = controls
    state["figure"] = fig
    state["axes"] = ax
    return state


def fft_image(image):
    """Compute a centered 2D Fourier transform.

    Parameters
    ----------
    image : array-like
        Input grayscale or RGB image. RGB images are converted to grayscale.

    Returns
    -------
    F : ndarray, shape (H, W)
        Centered complex Fourier transform.
    """
    return np.fft.fftshift(np.fft.fft2(rgb_to_gray(image)))


def ifft_image(F):
    """Invert a centered 2D Fourier transform.

    Parameters
    ----------
    F : array-like, shape (H, W)
        Centered complex Fourier transform.

    Returns
    -------
    image : ndarray, shape (H, W)
        Reconstructed real-valued image.
    """
    return np.real(np.fft.ifft2(np.fft.ifftshift(F)))


def fft_magnitude_phase(image):
    """Compute Fourier magnitude and phase.

    Parameters
    ----------
    image : array-like
        Input image.

    Returns
    -------
    magnitude, phase : tuple[ndarray, ndarray]
        Magnitude and phase of the centered Fourier transform.
    """
    F = fft_image(image)
    return np.abs(F), np.angle(F)


def parse_exposure_pattern(bits):
    """Convert a binary exposure string or sequence into a float array."""
    if isinstance(bits, str):
        bits = bits.strip()
        if not bits or set(bits) - {"0", "1"}:
            raise ValueError("Exposure strings must contain only 0 and 1.")
        return np.array([int(bit) for bit in bits], dtype=float)
    pattern = np.asarray(bits, dtype=float)
    if pattern.ndim != 1 or pattern.size == 0:
        raise ValueError("Exposure patterns must be non-empty 1D sequences.")
    if not np.all(np.isin(pattern, [0, 1])):
        raise ValueError("Exposure patterns must contain only 0 and 1.")
    return pattern


def make_exposure_kernel(pattern):
    """Convert a binary exposure schedule into a physical-throughput PSF."""
    pattern = parse_exposure_pattern(pattern)
    return pattern[None, :] / pattern.size


def make_gaussian_point_image(shape=(33, 64), center=None, sigma=1.2):
    """Create a visible approximation to a point on a black image."""
    height, width = map(int, shape)
    if center is None:
        center = (height // 2, width // 2)
    center_y, center_x = center
    yy, xx = np.mgrid[:height, :width]
    sigma = max(float(sigma), 1e-6)
    return np.exp(
        -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma**2)
    )


def motion_point_response(point_image, kernel):
    """Accumulate shifted copies of a point image according to a 1D motion PSF."""
    point = np.asarray(point_image, dtype=float)
    weights = np.asarray(kernel, dtype=float).ravel()
    offsets = np.arange(weights.size) - weights.size // 2
    response = np.zeros_like(point)
    for offset, weight in zip(offsets, weights):
        response += weight * np.roll(point, shift=offset, axis=1)
    return response


def show_exposure_schedule_point_responses(point_image, kernels, response_vmax=None):
    """Align exposure schedules above their captured point responses.

    Parameters
    ----------
    point_image : array-like, shape (H, W)
        Sharp point image used to visualize each PSF.
    kernels : mapping[str, array-like]
        Ordered mapping from display names to horizontal motion kernels.
    response_vmax : float or None
        Shared response display maximum. If None, infer it from the cases.

    Returns
    -------
    responses : dict[str, ndarray]
        Captured point response for each supplied kernel.
    """
    kernels = {name: np.asarray(kernel, dtype=float) for name, kernel in kernels.items()}
    responses = {
        name: motion_point_response(point_image, kernel)
        for name, kernel in kernels.items()
    }
    if response_vmax is None:
        response_vmax = max(response.max() for response in responses.values())

    num_cases = len(kernels)
    fig, ax = plt.subplots(
        2,
        1 + num_cases,
        figsize=(4 + 3.5 * num_cases, 6),
        dpi=130,
        gridspec_kw={"width_ratios": [0.85] + [1] * num_cases},
    )
    ax[0, 0].imshow(
        point_image, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
    )
    ax[0, 0].set_title("Sharp input\nsmall Gaussian point")
    ax[0, 0].set_axis_off()
    ax[1, 0].text(
        0.5,
        0.5,
        "exposure schedule\n" + r"$\downarrow$" + "\ncaptured point",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax[1, 0].set_axis_off()

    for col, (name, kernel) in enumerate(kernels.items(), start=1):
        weights = kernel.ravel()
        ax[0, col].stem(np.arange(weights.size), weights, basefmt=" ")
        ax[0, col].set_title(f"{name}\nthroughput={kernel.sum():.2f}")
        ax[0, col].set_xlabel("motion position")
        ax[0, col].set_ylim(0, max(0.08, 1.15 * weights.max()))

        ax[1, col].imshow(
            responses[name],
            cmap="gray",
            vmin=0,
            vmax=response_vmax,
            interpolation="nearest",
        )
        ax[1, col].set_title(f"{name} capture")
        ax[1, col].set_xlabel("horizontal sensor position")
        ax[1, col].set_yticks([])

    ax[0, 1].set_ylabel("exposure weight")
    plt.tight_layout(h_pad=2.0)
    plt.show()
    return responses


def show_frequency_responses(otfs, frequencies=None):
    """Plot relative horizontal frequency responses for several OTFs."""
    otfs = {name: np.asarray(H) for name, H in otfs.items()}
    first = next(iter(otfs.values()))
    if frequencies is None:
        frequencies = np.fft.fftshift(np.fft.fftfreq(first.shape[1]))
    ncols = 2
    nrows = int(np.ceil(len(otfs) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(11, 3.5 * nrows), dpi=130, sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    for axis, (name, H) in zip(axes, otfs.items()):
        throughput = float(np.abs(H[0, 0]))
        if throughput <= 0:
            raise ValueError(f"{name} has zero throughput.")
        relative_response = np.fft.fftshift(np.abs(H[0]) / throughput)
        axis.semilogy(frequencies, relative_response + 1e-6)
        axis.set_title(f"{name}: throughput H(0)={throughput:.3f}")
        axis.set_xlabel("horizontal frequency (cycles/pixel)")
        axis.set_ylabel("relative magnitude |H| / H(0)")
        axis.set_ylim(1e-4, 1.2)
        axis.grid(alpha=0.25)
    for axis in axes[len(otfs) :]:
        axis.set_axis_off()
    plt.tight_layout(h_pad=2.0)
    plt.show()
    return fig, axes


def simulate_photon_capture(expected_image, photons_per_unit_intensity=2000, rng=None):
    """Sample a simple Poisson photon-counting observation."""
    if rng is None:
        rng = np.random.default_rng()
    expected_counts = float(photons_per_unit_intensity) * np.clip(
        np.asarray(expected_image, dtype=float), 0, None
    )
    measured_counts = rng.poisson(expected_counts)
    return measured_counts / float(photons_per_unit_intensity)


def show_image_case_grid(
    source,
    cases,
    titles=None,
    source_title="Sharp source",
    suptitle=None,
    vmin=0,
    vmax=1,
):
    """Show a source image followed by an ordered mapping of image cases."""
    cases = dict(cases)
    titles = {} if titles is None else dict(titles)
    num_panels = 1 + len(cases)
    ncols = min(3, num_panels)
    nrows = int(np.ceil(num_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), dpi=130)
    axes = np.atleast_1d(axes).ravel()
    axes[0].imshow(source, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(source_title)
    axes[0].set_axis_off()
    for axis, (name, image) in zip(axes[1:], cases.items()):
        axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(titles.get(name, name))
        axis.set_axis_off()
    for axis in axes[num_panels:]:
        axis.set_axis_off()
    if suptitle:
        fig.suptitle(suptitle, y=1.01)
    plt.tight_layout(h_pad=2.0)
    plt.show()
    return fig, axes


def show_regularization_sweep(
    source,
    observations,
    otfs,
    regularization_values,
    reconstruction_function,
):
    """Apply and display one reconstruction method across regularization values."""
    observations = dict(observations)
    otfs = dict(otfs)
    if observations.keys() != otfs.keys():
        raise ValueError("Observations and OTFs must have matching case names.")

    values = list(regularization_values)
    fig, axes = plt.subplots(
        len(observations),
        len(values),
        figsize=(3.5 * len(values), 3.5 * len(observations)),
        dpi=130,
        squeeze=False,
    )
    reconstructions = {name: {} for name in observations}
    errors = {name: {} for name in observations}
    for row, name in enumerate(observations):
        for col, regularization in enumerate(values):
            reconstruction = reconstruction_function(
                observations[name], otfs[name], regularization
            )
            mse = float(np.mean((source - reconstruction) ** 2))
            reconstructions[name][regularization] = reconstruction
            errors[name][regularization] = mse
            axes[row, col].imshow(
                reconstruction, cmap="gray", vmin=0, vmax=1
            )
            axes[row, col].set_title(
                f"lambda={regularization:.0e}\nMSE={mse:.4f}"
            )
            axes[row, col].set_axis_off()
        axes[row, 0].set_ylabel(name)

    plt.tight_layout(h_pad=2.0)
    plt.show()
    return reconstructions, errors


def coded_exposure_frequency_widget(image, otfs, value=16, max_cycles=None):
    """Interactively probe how several exposure OTFs retain a sinusoid."""
    img = np.asarray(image, dtype=float)
    height, width = img.shape
    otfs = {name: np.asarray(H) for name, H in otfs.items()}
    if len(otfs) > 5:
        raise ValueError("The frequency probe displays at most five OTFs.")
    if max_cycles is None:
        max_cycles = width // 2
    control = IntSlider(
        min=0,
        max=int(max_cycles),
        step=1,
        value=int(np.clip(value, 0, max_cycles)),
        description="cycles/image",
    )
    frequencies = np.fft.fftshift(np.fft.fftfreq(width))

    def draw(cycles_per_image):
        x = np.arange(width)
        row = 0.5 + 0.5 * np.cos(2 * np.pi * cycles_per_image * x / width)
        grating = np.tile(row, (height, 1))
        frequency = cycles_per_image / width

        fig = plt.figure(figsize=(12, 9.5), dpi=120)
        grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.15])
        image_axes = [
            fig.add_subplot(grid[row_index, col_index])
            for row_index in range(2)
            for col_index in range(3)
        ]
        image_axes[0].imshow(grating, cmap="gray", vmin=0, vmax=1)
        image_axes[0].set_title(f"Input: {cycles_per_image} cycles/image")
        image_axes[0].set_axis_off()

        for axis, (name, H) in zip(image_axes[1:], otfs.items()):
            output = np.real(np.fft.ifft2(np.fft.fft2(grating) * H))
            throughput = np.abs(H[0, 0])
            selected = np.abs(H[0, cycles_per_image]) / throughput
            axis.imshow(output, cmap="gray", vmin=0, vmax=1)
            axis.set_title(
                f"{name}: relative={selected:.3f}\nthroughput={throughput:.3f}"
            )
            axis.set_axis_off()

        for axis in image_axes[1 + len(otfs) :]:
            axis.set_axis_off()

        response_axis = fig.add_subplot(grid[2, :])
        for name, H in otfs.items():
            throughput = np.abs(H[0, 0])
            response = np.fft.fftshift(np.abs(H[0]) / throughput)
            response_axis.semilogy(frequencies, response + 1e-6, label=name)
        response_axis.axvline(frequency, color="black", linestyle="--")
        response_axis.axvline(-frequency, color="black", linestyle="--")
        response_axis.set_title(
            "Relative frequency responses; dashed line is selected frequency"
        )
        response_axis.set_xlabel("horizontal frequency (cycles/pixel)")
        response_axis.set_ylabel("relative magnitude |H| / H(0)")
        response_axis.set_ylim(1e-4, 1.2)
        response_axis.grid(alpha=0.25)
        response_axis.legend(ncol=3, fontsize=8)
        fig.suptitle("Same input frequency, same intensity scale", y=0.98)
        fig.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.08,
            top=0.92,
            hspace=0.48,
            wspace=0.14,
        )
        plt.show()

    output = interactive_output(draw, {"cycles_per_image": control})
    display(VBox([control, output]))
    return {"cycles_per_image": control}


def make_frequency_mask(shape, kind="lowpass", radius=30, inner_radius=10, softness=0):
    """Create a circular Fourier-domain mask.

    Parameters
    ----------
    shape : tuple[int, int]
        Mask shape as (height, width).
    kind : {"lowpass", "highpass", "bandpass"}
        Frequency mask type.
    radius : float
        Outer radius in frequency pixels.
    inner_radius : float
        Inner radius for band-pass masks.
    softness : float
        Edge softness. Zero gives a hard binary mask.

    Returns
    -------
    mask : ndarray, shape (H, W)
        Frequency-domain mask with values in [0, 1].
    """
    h, w = shape
    yy, xx = np.indices((h, w))
    distance = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
    radius = max(float(radius), 1e-6)
    softness = max(float(softness), 0.0)

    def smooth_less_than(r):
        if softness == 0:
            return (distance <= r).astype(float)
        return 1 / (1 + np.exp((distance - r) / softness))

    low = smooth_less_than(radius)
    if kind == "lowpass":
        return low
    if kind == "highpass":
        return 1 - low
    if kind == "bandpass":
        inner = smooth_less_than(inner_radius)
        return np.clip(low - inner, 0, 1)
    raise ValueError("kind must be 'lowpass', 'highpass', or 'bandpass'")


def frequency_filter(image, mask):
    """Filter an image by multiplying its Fourier transform by a mask.

    Parameters
    ----------
    image : array-like
        Input image.
    mask : array-like, shape (H, W)
        Frequency-domain mask.

    Returns
    -------
    filtered : ndarray
        Real-valued filtered image.
    """
    F = fft_image(image)
    return ifft_image(F * np.asarray(mask))


def make_hybrid_image(
    low_image,
    high_image,
    low_sigma=6.0,
    high_sigma=2.0,
    high_weight=1.5,
    high_translation=(0, 0),
):
    """Combine low frequencies from one image with high frequencies from another.

    Parameters
    ----------
    low_image : array-like
        Image contributing low frequencies.
    high_image : array-like
        Image contributing high frequencies.
    low_sigma : float
        Gaussian sigma for low-pass filtering.
    high_sigma : float
        Gaussian sigma used to remove low frequencies from high_image.
    high_weight : float
        Weight on the high-frequency component.
    high_translation : tuple[int, int]
        Translation (tx, ty) applied to high_image before filtering.

    Returns
    -------
    hybrid, low_part, high_part : tuple[ndarray, ndarray, ndarray]
        Hybrid image and its two components.
    """
    low = rgb_to_gray(low_image)
    high = rgb_to_gray(high_image)
    shape = (min(low.shape[0], high.shape[0]), min(low.shape[1], high.shape[1]))
    if low.shape != shape:
        low = resize_image(low, output_shape=shape)
    if high.shape != shape:
        high = resize_image(high, output_shape=shape)
    high = translate_image(high, tx=high_translation[0], ty=high_translation[1], fill=float(np.mean(high)))
    low_part = gaussian_blur(low, sigma=low_sigma)
    high_part = high - gaussian_blur(high, sigma=high_sigma)
    hybrid = np.clip(low_part + high_weight * high_part, 0, 1)
    return hybrid, low_part, high_part


def sample_image(image, step=4):
    """Subsample an image by keeping every step-th pixel.

    Parameters
    ----------
    image : array-like
        Input image.
    step : int
        Sampling stride in pixels.

    Returns
    -------
    sampled : ndarray
        Subsampled image.
    """
    step = max(1, int(step))
    return as_float_image(image)[::step, ::step]


def anti_alias_downsample(image, step=4, sigma=None):
    """Blur and then subsample an image.

    Parameters
    ----------
    image : array-like
        Input image.
    step : int
        Downsampling stride.
    sigma : float or None
        Blur sigma. If None, use step / 2.

    Returns
    -------
    downsampled : ndarray
        Anti-aliased downsampled image.
    """
    step = max(1, int(step))
    sigma = step / 2 if sigma is None else sigma
    return gaussian_blur(image, sigma=sigma)[::step, ::step]


def sinusoid_pattern(size=256, frequency=12, angle_degrees=0):
    """Create a sinusoidal test pattern.

    Parameters
    ----------
    size : int
        Output width and height.
    frequency : float
        Number of cycles across the image.
    angle_degrees : float
        Stripe orientation control.

    Returns
    -------
    pattern : ndarray, shape (size, size)
        Sinusoidal pattern with values in [0, 1].
    """
    y, x = np.indices((size, size), dtype=float)
    theta = np.deg2rad(angle_degrees)
    coord = np.cos(theta) * x + np.sin(theta) * y
    return 0.5 + 0.5 * np.sin(2 * np.pi * frequency * coord / size)


def checkerboard_pattern(size=256, squares=24):
    """Create a checkerboard test pattern.

    Parameters
    ----------
    size : int
        Output width and height.
    squares : int
        Number of square periods across the image.

    Returns
    -------
    pattern : ndarray, shape (size, size)
        Checkerboard pattern with values 0 and 1.
    """
    y, x = np.indices((size, size))
    width = max(1, size // int(squares))
    return ((x // width + y // width) % 2).astype(float)


def _kernel_canvas(kernel, canvas_shape=(41, 41)):
    kernel = np.asarray(kernel, dtype=float)
    canvas = np.zeros(canvas_shape, dtype=float)
    h, w = kernel.shape
    y0 = canvas_shape[0] // 2 - h // 2
    x0 = canvas_shape[1] // 2 - w // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = kernel
    extent = (x0 - 0.5, x0 + w - 0.5, y0 + h - 0.5, y0 - 0.5)
    return canvas, extent


def _small_image(image, max_side=420, grayscale=False):
    img = rgb_to_gray(image) if grayscale else as_float_image(image)
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    return resize_image(img, scale=scale) if scale < 1 else img


def kernel_playground(image):
    """Display an editable 3x3 kernel playground.

    Parameters
    ----------
    image : array-like
        Input image.

    Returns
    -------
    state : dict
        Latest input image, kernel, filtered image, and widget controls.
    """
    img = _small_image(image, max_side=360, grayscale=True)
    controls = {}
    state = {"input": img, "kernel": None, "filtered": None, "controls": controls}
    defaults = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    for r in range(3):
        for c in range(3):
            controls[f"k{r}{c}"] = FloatSlider(
                min=-2,
                max=2,
                step=0.05,
                value=float(defaults[r, c]),
                description=f"k{r}{c}",
                readout_format=".2f",
            )

    def draw(**values):
        kernel = np.array([[values[f"k{r}{c}"] for c in range(3)] for r in range(3)], dtype=float)
        filtered = convolve_image(img, kernel)
        state["kernel"] = kernel
        state["filtered"] = filtered
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
        _show(img, f"Input\nmean={img.mean():.2f}", ax[0], vmin=0, vmax=1)
        ax[1].imshow(kernel, cmap="coolwarm")
        ax[1].set_title(f"Kernel, sum = {kernel.sum():.2f}")
        for r in range(3):
            for c in range(3):
                ax[1].text(c, r, f"{kernel[r, c]:.2f}", ha="center", va="center")
        ax[1].set_axis_off()
        _show(filtered, f"Filtered output\nmean={filtered.mean():.2f}", ax[2], vmin=0, vmax=1)
        plt.tight_layout()
        plt.show()

    rows = [HBox([controls[f"k{r}{c}"] for c in range(3)]) for r in range(3)]
    out = interactive_output(draw, controls)
    display(VBox(rows), out)
    return state


def blur_comparison_widget(image):
    """Compare box and Gaussian blur parameters interactively.

    Parameters
    ----------
    image : array-like
        Input image.

    Returns
    -------
    controls : dict
        Widget controls for blur type, kernel size, and sigma.
    """
    img = _small_image(image, max_side=420)
    controls = {
        "filter_type": Dropdown(options=["box", "gaussian"], value="gaussian", description="filter"),
        "size_x": IntSlider(min=1, max=41, step=2, value=15, description="size x"),
        "size_y": IntSlider(min=1, max=41, step=2, value=15, description="size y"),
        "sigma_x": FloatSlider(min=0.3, max=10, step=0.1, value=2.5, description="sigma x", readout_format=".1f"),
        "sigma_y": FloatSlider(min=0.3, max=10, step=0.1, value=2.5, description="sigma y", readout_format=".1f"),
    }

    def draw(filter_type, size_x, size_y, sigma_x, sigma_y):
        size = (size_y, size_x)
        sigma = (sigma_y, sigma_x)
        kernel = make_box_filter(size) if filter_type == "box" else make_gaussian_filter(size, sigma)
        blurred = np.clip(convolve_image(img, kernel), 0, 1)
        residual = np.clip(0.5 + 2 * (img - blurred), 0, 1)
        canvas, extent = _kernel_canvas(kernel)
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.5))
        _show(img, "Input", ax[0])
        ax[1].imshow(canvas, cmap="viridis", vmin=0, vmax=max(canvas.max(), 1e-12))
        ax[1].plot(
            [extent[0], extent[1], extent[1], extent[0], extent[0]],
            [extent[2], extent[2], extent[3], extent[3], extent[2]],
            color="white",
            linewidth=1.5,
        )
        ax[1].set_title(f"Kernel support: {size_x} x {size_y}")
        ax[1].set_axis_off()
        _show(blurred, "Blurred", ax[2])
        _show(residual, "Amplified residual", ax[3])
        plt.tight_layout()
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def matrix_gaussian_explorer():
    """Explore a 2x2 covariance matrix and the Gaussian it defines."""
    controls = {
        "sigma_x": FloatSlider(min=0.5, max=8, step=0.1, value=5, description="sigma x", readout_format=".1f"),
        "sigma_y": FloatSlider(min=0.5, max=8, step=0.1, value=2, description="sigma y", readout_format=".1f"),
        "cov_xy": FloatSlider(min=-25, max=25, step=0.5, value=0, description="cov xy", readout_format=".1f"),
    }

    def draw(sigma_x, sigma_y, cov_xy):
        covariance = np.array([[sigma_x**2, cov_xy], [cov_xy, sigma_y**2]], dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].axis("off")
        ax[0].text(
            0.05,
            0.75,
            f"Sigma = [[{covariance[0, 0]:.2f}, {covariance[0, 1]:.2f}],\n"
            f"         [{covariance[1, 0]:.2f}, {covariance[1, 1]:.2f}]]",
            fontsize=13,
            family="monospace",
        )

        if np.any(eigenvalues <= 0):
            ax[1].axis("off")
            ax[1].text(
                0.5,
                0.5,
                "not a valid covariance matrix\n"
                "has to be positive semidefinite\n"
                "https://en.wikipedia.org/wiki/Positive-semidefinite_matrix",
                ha="center",
                va="center",
                fontsize=11,
            )
            plt.show()
            return

        radius = 40
        y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        points = np.stack([x.ravel(), y.ravel()], axis=1)
        inv_covariance = np.linalg.inv(covariance)
        values = np.exp(-0.5 * np.sum((points @ inv_covariance) * points, axis=1))
        gaussian = values.reshape(2 * radius + 1, 2 * radius + 1)

        ax[1].imshow(gaussian, cmap="viridis", extent=(-radius, radius, radius, -radius))
        ax[1].set_title("Gaussian from Sigma")
        ax[1].set_aspect("equal")
        ax[1].set_axis_off()
        plt.tight_layout()
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox([HBox([controls["sigma_x"], controls["sigma_y"]]), controls["cov_xy"]]), out)
    return controls


def covariance_explorer():
    """Explore eigenvectors of a rotated 2D covariance matrix."""
    controls = {
        "sigma_1": FloatSlider(min=0.5, max=8, step=0.1, value=5, description="sigma 1", readout_format=".1f"),
        "sigma_2": FloatSlider(min=0.5, max=8, step=0.1, value=2, description="sigma 2", readout_format=".1f"),
        "angle": FloatSlider(min=-90, max=90, step=1, value=30, description="angle", readout_format=".0f"),
        "vector_angle": FloatSlider(min=-180, max=180, step=1, value=0, description="vector", readout_format=".0f"),
    }

    def draw(sigma_1, sigma_2, angle, vector_angle):
        theta = np.deg2rad(angle)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        covariance = R @ np.diag([sigma_1**2, sigma_2**2]) @ R.T
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        t = np.linspace(0, 2 * np.pi, 300)
        circle = np.stack([np.cos(t), np.sin(t)], axis=1)
        ellipse = circle @ np.diag([sigma_1, sigma_2]) @ R.T

        phi = np.deg2rad(vector_angle)
        v = np.array([np.cos(phi), np.sin(phi)])
        transformed = covariance @ v
        transformed = transformed / max(eigenvalues) * max(sigma_1, sigma_2)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(ellipse[:, 0], ellipse[:, 1], color="black", linewidth=2)
        ax.axhline(0, color="0.85")
        ax.axvline(0, color="0.85")
        ax.arrow(
            0,
            0,
            v[0] * max(sigma_1, sigma_2),
            v[1] * max(sigma_1, sigma_2),
            color="tab:blue",
            width=0.03,
            length_includes_head=True,
            label=r"$v$",
        )
        ax.arrow(
            0,
            0,
            transformed[0],
            transformed[1],
            color="tab:red",
            width=0.03,
            length_includes_head=True,
            label=r"$\Sigma v$",
        )
        for i in range(2):
            direction = eigenvectors[:, i]
            length = np.sqrt(eigenvalues[i])
            ax.plot(
                [-direction[0] * length, direction[0] * length],
                [-direction[1] * length, direction[1] * length],
                color="tab:green",
                linewidth=3,
                label="eigenvectors" if i == 0 else None,
            )
        ax.set_title(r"$\Sigma v = \lambda v$")
        ax.legend(loc="upper right")
        limit = max(sigma_1, sigma_2) * 1.4
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox([HBox([controls["sigma_1"], controls["sigma_2"]]), HBox([controls["angle"], controls["vector_angle"]])]), out)
    return controls


def gradient_widget(image):
    """Explore derivative filters and gradient magnitude.

    Parameters
    ----------
    image : array-like
        Input image.

    Returns
    -------
    controls : dict
        Widget controls for derivative method, smoothing, and display mode.
    """
    img = _small_image(image, max_side=420, grayscale=True)
    controls = {
        "method": Dropdown(options=["sobel", "finite"], value="sobel", description="method"),
        "sigma": FloatSlider(min=0, max=5, step=0.2, value=1.0, description="smooth", readout_format=".1f"),
        "threshold": FloatSlider(min=0, max=0.5, step=0.01, value=0.08, description="threshold", readout_format=".2f"),
    }

    def draw(method, sigma, threshold):
        Ix, Iy, mag, _ = image_gradients(img, sigma=sigma, method=method)
        edges = mag > threshold
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.5))
        _show(Ix, "Ix", ax[0], cmap="coolwarm")
        _show(Iy, "Iy", ax[1], cmap="coolwarm")
        _show(mag, "Gradient magnitude", ax[2], cmap="gray")
        _show(edges, "Thresholded edges", ax[3], cmap="gray")
        plt.tight_layout()
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def sharpening_widget(image):
    """Explore unsharp masking parameters.

    Parameters
    ----------
    image : array-like
        Input image.

    Returns
    -------
    controls : dict
        Widget controls for blur radius and sharpening amount.
    """
    img = _small_image(image, max_side=420)
    controls = {
        "radius": FloatSlider(min=0.5, max=8, step=0.1, value=2.0, description="radius", readout_format=".1f"),
        "amount": FloatSlider(min=0, max=4, step=0.1, value=1.2, description="amount", readout_format=".1f"),
    }

    def draw(radius, amount):
        sharp, blurred, detail = unsharp_mask(
            img,
            radius=radius,
            amount=amount,
        )

        added_detail = amount * detail

        fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
        ax = ax.ravel()

        _show(
            img,
            r"Original $I$",
            ax[0],
        )

        _show(
            blurred,
            rf"Blurred $I_{{\mathrm{{blur}}}} = I * k_{{\mathrm{{blur}}}}$"
            f"\n`radius` = {radius:.1f}",
            ax[1],
        )

        # Zero contribution is displayed as middle gray.
        added_detail_display = np.clip(0.5 + added_detail, 0, 1)
        _show(
            added_detail_display,
            rf"Added term $\alpha I_{{\mathrm{{detail}}}}$"
            f"\n`amount` = $\\alpha$ = {amount:.1f}; zero is gray",
            ax[2],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        _show(
            sharp,
            r"Sharpened $I_{\mathrm{sharp}}"
            r" = I + \alpha I_{\mathrm{detail}}$",
            ax[3],
        )

        plt.tight_layout(h_pad=3.0)
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def bilateral_widget(image):
    """Compare Gaussian and bilateral smoothing.

    Parameters
    ----------
    image : array-like
        Input image.

    Returns
    -------
    controls : dict
        Widget controls for bilateral filter parameters.
    """
    img = _small_image(image, max_side=140)
    controls = {
        "spatial_sigma": FloatSlider(min=1, max=8, step=0.5, value=3, description="spatial", readout_format=".1f"),
        "range_sigma": FloatSlider(min=0.02, max=0.5, step=0.01, value=0.12, description="range", readout_format=".2f"),
    }

    def draw(spatial_sigma, range_sigma):
        gaussian = gaussian_blur(img, sigma=spatial_sigma)
        bilateral = bilateral_filter(img, spatial_sigma=spatial_sigma, range_sigma=range_sigma)
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.5))
        _show(img, "Input crop", ax[0])
        _show(gaussian, "Gaussian", ax[1])
        _show(bilateral, "Bilateral", ax[2])
        plt.tight_layout()
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def interpolation_widget(image):
    """Explore resizing and interpolation methods.

    Parameters
    ----------
    image : array-like
        Input image.

    Returns
    -------
    controls : dict
        Widget controls for crop position, zoom, and interpolation.
    """
    img = as_float_image(image)
    h, w = img.shape[:2]
    controls = {
        "x": IntSlider(min=0, max=max(0, w - 120), step=5, value=max(0, w // 3), description="x"),
        "y": IntSlider(min=0, max=max(0, h - 120), step=5, value=max(0, h // 3), description="y"),
        "zoom": FloatSlider(min=1, max=8, step=0.5, value=4, description="zoom", readout_format=".1f"),
        "method": Dropdown(options=["nearest", "bilinear", "bicubic", "lanczos"], value="nearest", description="method"),
    }

    def draw(x, y, zoom, method):
        crop = img[y : min(h, y + 120), x : min(w, x + 120)]
        enlarged = resize_image(crop, scale=zoom, method=method)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        _show(crop, "Original crop", ax[0])
        _show(enlarged, f"{method}, zoom {zoom:.1f}x", ax[1])
        plt.tight_layout()
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def _single_frequency_pair(F, k):
    pair = np.zeros_like(F)
    pair[k] = F[k]
    pair[-k] = F[-k]
    return pair


def fourier_1d_reconstruction_widget():
    """Explore 1D Fourier reconstruction with strongest frequency pairs."""
    controls = {
        "signal": Dropdown(options=["square", "step", "two spikes", "smooth"], value="square", description="signal"),
        "components": IntSlider(min=0, max=80, step=1, value=0, description="components"),
    }
    n = 256
    x = np.linspace(0, 1, n, endpoint=False)

    def make_signal(kind):
        if kind == "square":
            return (np.sin(2 * np.pi * 4 * x) > 0).astype(float)
        if kind == "step":
            return (x > 0.38).astype(float)
        if kind == "two spikes":
            y = np.zeros_like(x)
            y[55:70] = 1
            y[150:158] = 0.75
            return y
        return 0.45 + 0.35 * np.sin(2 * np.pi * 3 * x) + 0.15 * np.sin(2 * np.pi * 9 * x)

    def draw(signal, components):
        y = make_signal(signal)
        F = np.fft.fft(y)
        mask = np.zeros(n, dtype=float)
        selected = np.array([], dtype=int)
        if components > 0:
            mask[0] = 1
        if components > 1:
            pair_scores = np.abs(F[1 : n // 2])
            selected = np.argsort(pair_scores)[-(components - 1) :] + 1
            mask[selected] = 1
            mask[-selected] = 1
        reconstruction = np.real(np.fft.ifft(F * mask))
        residual = y - reconstruction

        fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
        ax = ax.ravel()
        ax[0].plot(x, y, color="black")
        ax[0].set_title("signal")
        ax[1].plot(x, reconstruction, color="tab:blue")
        ax[1].set_title(f"reconstruction: components = {components}")
        ax[2].axhline(0, color="0.85")
        if components > 0:
            dc = np.real(F[0] / n)
            ax[2].plot(x, np.full_like(x, dc), color="0.4", linewidth=2, label="DC")
        for k in selected:
            component = np.real(np.fft.ifft(_single_frequency_pair(F, k)))
            ax[2].plot(x, component, linewidth=1)
        ax[2].set_title("added components")
        ax[3].plot(x, residual, color="tab:red")
        ax[3].axhline(0, color="0.8")
        ax[3].set_title("residual")
        for axis in ax[:2]:
            axis.set_xlim(0, 1)
            axis.set_ylim(-0.4, 1.4)
        for axis in ax[2:]:
            axis.set_xlim(0, 1)
        plt.tight_layout()
        plt.show()

    def reset_components_on_signal_change(change):
        if change["name"] == "value":
            controls["components"].value = 0

    controls["signal"].observe(reset_components_on_signal_change, names="value")
    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def fourier_radius_reconstruction_widget(image_a, image_b, title_a="image A", title_b="image B"):
    """Reconstruct two images using Fourier coefficients within a radius."""
    img_a = resize_image(rgb_to_gray(image_a), output_shape=(256, 256))
    img_b = resize_image(rgb_to_gray(image_b), output_shape=(256, 256))
    images = [(img_a, title_a), (img_b, title_b)]
    transforms = [fft_image(img_a), fft_image(img_b)]
    controls = {
        "radius": FloatSlider(min=0, max=180, step=2, value=0, description="radius", readout_format=".0f"),
    }

    def draw(radius):
        mask = make_frequency_mask(img_a.shape, kind="lowpass", radius=radius, softness=2)
        fig, ax = plt.subplots(2, 4, figsize=(12, 6.5), dpi=150)
        for row, ((img, title), F) in enumerate(zip(images, transforms)):
            reconstruction = ifft_image(F * mask)
            residual = np.abs(img - np.clip(reconstruction, 0, 1))
            _show(img, title, ax[row, 0], cmap="gray", vmin=0, vmax=1)
            _show(np.log1p(np.abs(F)), "Fourier magnitude", ax[row, 1], cmap="magma")
            _show(mask, "kept frequencies", ax[row, 2], cmap="gray", vmin=0, vmax=1)
            _show(np.clip(reconstruction, 0, 1), "reconstruction", ax[row, 3], cmap="gray", vmin=0, vmax=1)
            ax[row, 1].contour(mask, levels=[0.5], colors="white", linewidths=1)
        plt.tight_layout()
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def fourier_component_widget(image, title="image"):
    """Explore one conjugate pair of 2D Fourier components."""
    img = resize_image(rgb_to_gray(image), output_shape=(256, 256))
    F = fft_image(img)
    h, w = img.shape
    cy, cx = h // 2, w // 2
    controls = {
        "freq_x": IntSlider(min=-80, max=80, step=1, value=12, description="freq x"),
        "freq_y": IntSlider(min=-80, max=80, step=1, value=0, description="freq y"),
    }

    def draw(freq_x, freq_y):
        mask = np.zeros_like(img)
        y1, x1 = int(np.clip(cy + freq_y, 0, h - 1)), int(np.clip(cx + freq_x, 0, w - 1))
        y2, x2 = int(np.clip(cy - freq_y, 0, h - 1)), int(np.clip(cx - freq_x, 0, w - 1))
        mask[y1, x1] = 1
        mask[y2, x2] = 1
        if freq_x != 0 or freq_y != 0:
            component = ifft_image(F * mask)
        else:
            component = np.full_like(img, np.real(F[cy, cx]) / (h * w))

        yy, xx = np.indices(img.shape)
        phase = 2 * np.pi * (freq_x * (xx - cx) / w + freq_y * (yy - cy) / h)
        grating = 0.5 + 0.5 * np.cos(phase)

        fig, ax = plt.subplots(1, 4, figsize=(13, 3.5))
        _show(img, title, ax[0], cmap="gray", vmin=0, vmax=1)
        _show(np.log1p(np.abs(F)), "Fourier magnitude", ax[1], cmap="magma")
        ax[1].plot([x1, x2], [y1, y2], "co", markersize=5)
        _show(np.angle(F), "Fourier phase", ax[2], cmap="twilight")
        ax[2].plot([x1, x2], [y1, y2], "ko", markersize=5)
        _show(grating, "matching sinusoid", ax[3], cmap="gray", vmin=0, vmax=1)
        plt.tight_layout()
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def fourier_phase_swap_demo(image_a, image_b, title_a="image A", title_b="image B"):
    """Swap Fourier magnitude and phase between two images."""
    a = resize_image(rgb_to_gray(image_a), output_shape=(256, 256))
    b = resize_image(rgb_to_gray(image_b), output_shape=(256, 256))
    Fa = fft_image(a)
    Fb = fft_image(b)

    a_mag_b_phase = ifft_image(np.abs(Fa) * np.exp(1j * np.angle(Fb)))
    b_mag_a_phase = ifft_image(np.abs(Fb) * np.exp(1j * np.angle(Fa)))

    def display_range(img):
        low, high = np.percentile(img, [1, 99])
        return np.clip((img - low) / (high - low + 1e-12), 0, 1)

    fig, ax = plt.subplots(2, 4, figsize=(13, 6.5))
    _show(a, title_a, ax[0, 0], cmap="gray", vmin=0, vmax=1)
    _show(np.log1p(np.abs(Fa)), f"{title_a} magnitude", ax[0, 1], cmap="magma")
    _show(np.angle(Fa), f"{title_a} phase", ax[0, 2], cmap="twilight")
    _show(display_range(a_mag_b_phase), f"{title_a} magnitude\n{title_b} phase", ax[0, 3], cmap="gray", vmin=0, vmax=1)
    _show(b, title_b, ax[1, 0], cmap="gray", vmin=0, vmax=1)
    _show(np.log1p(np.abs(Fb)), f"{title_b} magnitude", ax[1, 1], cmap="magma")
    _show(np.angle(Fb), f"{title_b} phase", ax[1, 2], cmap="twilight")
    _show(display_range(b_mag_a_phase), f"{title_b} magnitude\n{title_a} phase", ax[1, 3], cmap="gray", vmin=0, vmax=1)
    plt.tight_layout()
    plt.show()
    return {
        "image_a": a,
        "image_b": b,
        "a_magnitude_b_phase": a_mag_b_phase,
        "b_magnitude_a_phase": b_mag_a_phase,
    }


def fft_magnitude_phase_widget(image):
    """Explore Fourier magnitude and phase.

    Parameters
    ----------
    image : array-like
        Input image.

    Returns
    -------
    controls : dict
        Empty dictionary for consistency with other widgets.
    """
    img = _small_image(image, max_side=420, grayscale=True)
    F = fft_image(img)
    mag = np.abs(F)
    phase = np.angle(F)
    mag_only = ifft_image(mag)
    phase_only = ifft_image(np.exp(1j * phase))
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.5))
    _show(img, "Input", ax[0], vmin=0, vmax=1)
    _show(np.log1p(mag), "log magnitude", ax[1], cmap="magma")
    _show(phase, "phase", ax[2], cmap="twilight")
    _show(phase_only, "phase-only reconstruction", ax[3], cmap="gray")
    plt.tight_layout()
    plt.show()
    return {}


def frequency_filter_widget(image):
    """Explore Fourier-domain filtering masks.

    Parameters
    ----------
    image : array-like
        Input image.

    Returns
    -------
    controls : dict
        Widget controls for mask type and radius.
    """
    img = _small_image(image, max_side=420, grayscale=True)
    controls = {
        "kind": Dropdown(options=["lowpass", "highpass", "bandpass"], value="lowpass", description="mask"),
        "radius": FloatSlider(min=4, max=160, step=2, value=35, description="radius", readout_format=".0f"),
        "inner_radius": FloatSlider(min=2, max=100, step=2, value=12, description="inner", readout_format=".0f"),
        "softness": FloatSlider(min=0, max=20, step=1, value=3, description="soft", readout_format=".0f"),
    }

    def draw(kind, radius, inner_radius, softness):
        mask = make_frequency_mask(img.shape, kind=kind, radius=radius, inner_radius=inner_radius, softness=softness)
        filtered = frequency_filter(img, mask)
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.5))
        _show(img, "Input", ax[0], vmin=0, vmax=1)
        _show(np.log1p(np.abs(fft_image(img))), "Fourier magnitude", ax[1], cmap="magma")
        _show(mask, "Mask", ax[2], cmap="gray", vmin=0, vmax=1)
        _show(np.clip(filtered, 0, 1), "Filtered", ax[3], cmap="gray", vmin=0, vmax=1)
        plt.tight_layout()
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def convolution_theorem_widget(image):
    """Compare spatial convolution with Fourier multiplication.

    Parameters
    ----------
    image : array-like
        Input image.

    Returns
    -------
    controls : dict
        Widget controls for Gaussian sigma.
    """
    img = _small_image(image, max_side=360, grayscale=True)
    controls = {
        "sigma": FloatSlider(min=0.5, max=8, step=0.2, value=2.0, description="sigma", readout_format=".1f"),
    }

    def draw(sigma):
        spatial = gaussian_blur(img, sigma=sigma)
        impulse = np.zeros_like(img)
        impulse[img.shape[0] // 2, img.shape[1] // 2] = 1
        kernel_image = gaussian_blur(impulse, sigma=sigma)
        frequency = ifft_image(fft_image(img) * fft_image(kernel_image))
        diff = np.abs(spatial - frequency)
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.5))
        _show(spatial, "Spatial convolution", ax[0], cmap="gray", vmin=0, vmax=1)
        _show(np.log1p(np.abs(fft_image(kernel_image))), "Kernel spectrum", ax[1], cmap="magma")
        _show(np.clip(frequency, 0, 1), "Frequency multiplication", ax[2], cmap="gray", vmin=0, vmax=1)
        _show(diff, "Absolute difference", ax[3], cmap="gray")
        plt.tight_layout()
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def hybrid_image_widget(low_image, high_image, make_hybrid_fn=None):
    """Explore hybrid image parameters.

    Parameters
    ----------
    low_image : array-like
        Image contributing low frequencies.
    high_image : array-like
        Image contributing high frequencies.
    make_hybrid_fn : callable or None
        Function used to compute the hybrid image.

    Returns
    -------
    controls : dict
        Widget controls for low/high frequency parameters.
    """
    controls = {
        "low_sigma": FloatSlider(min=1, max=16, step=0.5, value=6, description="low sigma", readout_format=".1f", continuous_update=False),
        "high_sigma": FloatSlider(min=1, max=12, step=0.5, value=2.5, description="high sigma", readout_format=".1f", continuous_update=False),
        "high_weight": FloatSlider(min=0, max=4, step=0.1, value=1.5, description="weight", readout_format=".1f", continuous_update=False),
        "tx": IntSlider(min=-120, max=120, step=1, value=0, description="tx", continuous_update=False),
        "ty": IntSlider(min=-120, max=120, step=1, value=0, description="ty", continuous_update=False),
        "view_scale": FloatSlider(min=0.1, max=1.0, step=0.05, value=0.25, description="view scale", readout_format=".2f", continuous_update=False),
    }
    if make_hybrid_fn is None:
        make_hybrid_fn = make_hybrid_image
    low = _small_image(low_image, max_side=360, grayscale=True)
    high = _small_image(high_image, max_side=360, grayscale=True)
    shape = (min(low.shape[0], high.shape[0]), min(low.shape[1], high.shape[1]))
    low = resize_image(low, output_shape=shape)
    high = resize_image(high, output_shape=shape)

    def draw(low_sigma, high_sigma, high_weight, tx, ty, view_scale):
        hybrid, low_part, high_part = make_hybrid_fn(
            low,
            high,
            low_sigma=low_sigma,
            high_sigma=high_sigma,
            high_weight=high_weight,
            high_translation=(tx, ty),
        )
        small = resize_image(hybrid, scale=view_scale)
        far_view = np.full_like(hybrid, 0.5)
        y0 = (far_view.shape[0] - small.shape[0]) // 2
        x0 = (far_view.shape[1] - small.shape[1]) // 2
        far_view[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.5))
        _show(low_part, "Low frequencies", ax[0], cmap="gray", vmin=0, vmax=1)
        _show(np.clip(0.5 + 2 * high_part, 0, 1), "High frequencies", ax[1], cmap="gray", vmin=0, vmax=1)
        _show(hybrid, "Hybrid close", ax[2], cmap="gray", vmin=0, vmax=1)
        _show(far_view, "Hybrid small/far", ax[3], cmap="gray", vmin=0, vmax=1)
        plt.tight_layout()
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def sampling_widget():
    """Explore sampling and aliasing on synthetic patterns.

    Parameters
    ----------
    None

    Returns
    -------
    controls : dict
        Widget controls for pattern frequency and sample step.
    """
    controls = {
        "pattern": Dropdown(options=["sinusoid", "checkerboard"], value="sinusoid", description="pattern"),
        "frequency": FloatSlider(min=2, max=80, step=1, value=18, description="freq", readout_format=".0f"),
        "angle": FloatSlider(min=-90, max=90, step=5, value=0, description="angle", readout_format=".0f"),
        "step": IntSlider(min=1, max=16, step=1, value=4, description="step"),
    }

    def draw(pattern, frequency, angle, step):
        if pattern == "sinusoid":
            base = sinusoid_pattern(
                frequency=frequency,
                angle_degrees=angle,
            )
        else:
            base = checkerboard_pattern(
                squares=frequency,
            )

        sampled = sample_image(base, step=step)

        # Enlarge the lower-resolution sample array without smoothing it.
        shown_sampled = resize_image(
            sampled,
            output_shape=base.shape,
            method="nearest",
        )

        original_spectrum = np.log1p(
            np.abs(fft_image(base))
        )
        sampled_spectrum = np.log1p(
            np.abs(fft_image(shown_sampled))
        )

        fig, ax = plt.subplots(
            1,
            4,
            figsize=(15, 3.8),
            dpi=140,
        )

        _show(
            base,
            "Original pattern\n"
            f"shape = {base.shape[1]} × {base.shape[0]}",
            ax[0],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        _show(
            original_spectrum,
            "Original spectrum",
            ax[1],
            cmap="magma",
        )

        _show(
            shown_sampled,
            "Sampled, then enlarged\n"
            f"sample shape = {sampled.shape[1]} × {sampled.shape[0]}",
            ax[2],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        _show(
            sampled_spectrum,
            "Sampled spectrum",
            ax[3],
            cmap="magma",
        )

        plt.tight_layout()
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls


def downsampling_widget(image):
    """Compare naive and anti-aliased downsampling.

    Parameters
    ----------
    image : array-like
        Input image.

    Returns
    -------
    controls : dict
        Widget controls for stride and prefilter strength.
    """
    img = _small_image(image, max_side=600)
    controls = {
        "step": IntSlider(min=2, max=12, step=1, value=4, description="step"),
        "sigma": FloatSlider(min=0, max=6, step=0.2, value=2, description="sigma", readout_format=".1f"),
    }

    def draw(step, sigma):
        naive = sample_image(img, step=step)
        anti = anti_alias_downsample(img, step=step, sigma=sigma)
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
        _show(img, "Input", ax[0])
        _show(resize_image(naive, output_shape=img.shape[:2], method="nearest"), "Naive downsample", ax[1])
        _show(resize_image(anti, output_shape=img.shape[:2], method="nearest"), "Blur then downsample", ax[2])
        plt.tight_layout()
        plt.show()

    out = interactive_output(draw, controls)
    display(VBox(list(controls.values())), out)
    return controls
