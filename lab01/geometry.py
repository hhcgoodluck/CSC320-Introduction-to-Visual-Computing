"""Reusable geometry functions for CSC320 Notebook 1.

The functions are intentionally compact and dependency-light so students can
inspect, reuse, and modify them in project milestones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


HIDDEN_AFFINE_TARGET_PARAMS = {
    "scale": (0.82, 1.18),
    "rotation_degrees": 25,
    "shear": (0.18, -0.06),
    "translation": (52, 18),
}


HIDDEN_AFFINE_CHALLENGE_PARAMS = {
    "scale": (1.25, 0.72),
    "rotation_degrees": 0,
    "shear": (0.34, -0.16),
    "translation": (-42, 36),
}


HIDDEN_PROJECTION_TARGET_PARAMS = {
    "focal_length": 875,
    "pitch": -12,
    "yaw": 16,
    "roll": -8,
    "camera_x": 0.7,
    "camera_y": -0.3,
    "camera_z": 0.6,
}


CHALLENGE_AFFINE_PARAMS = HIDDEN_AFFINE_CHALLENGE_PARAMS
PROJECTION_TARGET_PARAMS = HIDDEN_PROJECTION_TARGET_PARAMS


def to_homogeneous(points):
    """Convert 2D or 3D points to homogeneous coordinates.

    Parameters
    ----------
    points : array-like, shape (..., D)
        Cartesian points, where D is usually 2 or 3.

    Returns
    -------
    homogeneous : ndarray, shape (..., D + 1)
        Input points with a final coordinate of 1 appended.
    """
    pts = np.asarray(points, dtype=float)
    ones = np.ones((*pts.shape[:-1], 1), dtype=float)
    return np.concatenate([pts, ones], axis=-1)


def from_homogeneous(points, eps=1e-12):
    """Convert homogeneous points to Cartesian coordinates.

    Parameters
    ----------
    points : array-like, shape (..., D + 1)
        Homogeneous points.
    eps : float
        Small value used to avoid division by exact zero.

    Returns
    -------
    cartesian : ndarray, shape (..., D)
        Points divided by their final homogeneous coordinate.
    """
    pts = np.asarray(points, dtype=float)
    w = pts[..., -1:]
    w = np.where(np.abs(w) < eps, np.sign(w) * eps + (w == 0) * eps, w)
    return pts[..., :-1] / w


def make_affine_transform(
    scale=(1.0, 1.0),
    rotation_degrees=0.0,
    shear=(0.0, 0.0),
    translation=(0.0, 0.0),
):
    """Create a 3x3 affine transform matrix.

    Parameters
    ----------
    scale : tuple[float, float]
        Horizontal and vertical scale factors.
    rotation_degrees : float
        Counter-clockwise rotation angle in degrees.
    shear : float or tuple[float, float]
        Shear amounts as (shear_x, shear_y). A single float is treated as
        horizontal shear for backward compatibility.
    translation : tuple[float, float]
        Horizontal and vertical translation in pixels.

    Returns
    -------
    H : ndarray, shape (3, 3)
        Homogeneous affine transform mapping source points to destination points.
    """
    sx, sy = scale
    if np.isscalar(shear):
        shear_x, shear_y = float(shear), 0.0
    else:
        shear_x, shear_y = shear
    tx, ty = translation
    theta = np.deg2rad(rotation_degrees)
    c, s = np.cos(theta), np.sin(theta)
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=float)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    Sh = np.array([[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]], dtype=float)
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)
    return T @ R @ Sh @ S


def estimate_affine_transform(src_points, dst_points):
    """Estimate a 2D affine transform from at least three correspondences.

    All supplied correspondences are fit in a least-squares sense, so four
    corners of a projectively distorted quadrilateral generally cannot all be
    mapped exactly to a rectangle.
    """
    src = np.asarray(src_points, dtype=float)
    dst = np.asarray(dst_points, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or src.shape[0] < 3:
        raise ValueError(
            "src_points and dst_points must both have shape (N, 2), N >= 3"
        )

    count = src.shape[0]
    system = np.zeros((2 * count, 6), dtype=float)
    target = dst.reshape(-1)
    system[0::2, 0:2] = src
    system[0::2, 2] = 1
    system[1::2, 3:5] = src
    system[1::2, 5] = 1
    parameters, _, rank, _ = np.linalg.lstsq(system, target, rcond=None)
    if rank < 6:
        raise ValueError("The point configuration does not determine an affine transform")

    return np.array(
        [
            [parameters[0], parameters[1], parameters[2]],
            [parameters[3], parameters[4], parameters[5]],
            [0.0, 0.0, 1.0],
        ]
    )


def rectangle_geometry_metrics(points):
    """Measure how closely an ordered quadrilateral resembles a rectangle.

    Returns mean absolute errors in degrees for adjacent right angles and for
    the two pairs of opposite, nominally parallel edges. Lower is better.
    """
    points = np.asarray(points, dtype=float)
    if points.shape != (4, 2):
        raise ValueError("points must have shape (4, 2)")

    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1e-12):
        return {
            "right_angle_error_deg": np.nan,
            "parallel_error_deg": np.nan,
        }
    unit_edges = edges / lengths[:, None]

    right_angle_errors = []
    for index in range(4):
        cosine = np.clip(
            np.dot(unit_edges[index], unit_edges[(index + 1) % 4]), -1.0, 1.0
        )
        angle = np.degrees(np.arccos(cosine))
        right_angle_errors.append(abs(90.0 - angle))

    parallel_errors = []
    for first, second in ((0, 2), (1, 3)):
        cosine = np.clip(abs(np.dot(unit_edges[first], unit_edges[second])), 0.0, 1.0)
        parallel_errors.append(np.degrees(np.arccos(cosine)))

    return {
        "right_angle_error_deg": float(np.mean(right_angle_errors)),
        "parallel_error_deg": float(np.mean(parallel_errors)),
    }


def estimate_homography(src_points, dst_points):
    """Estimate a homography from point correspondences using DLT.

    Parameters
    ----------
    src_points : array-like, shape (N, 2)
        Source image coordinates.
    dst_points : array-like, shape (N, 2)
        Destination image coordinates.

    Returns
    -------
    H : ndarray, shape (3, 3)
        Homography matrix mapping source points to destination points.
    """
    src = np.asarray(src_points, dtype=float)
    dst = np.asarray(dst_points, dtype=float)
    if src.shape != dst.shape or src.shape[0] < 4 or src.shape[1] != 2:
        raise ValueError("src_points and dst_points must both have shape (N, 2), N >= 4")

    src_n, T_src = _normalize_points(src)
    dst_n, T_dst = _normalize_points(dst)
    A = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
    _, _, vh = np.linalg.svd(np.asarray(A))
    H_n = vh[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_n @ T_src
    return H / H[-1, -1]


def make_homography(src_points, dst_points):
    """Create a homography from at least four point correspondences.

    Parameters
    ----------
    src_points : array-like, shape (N, 2)
        Source coordinates.
    dst_points : array-like, shape (N, 2)
        Destination coordinates.

    Returns
    -------
    H : ndarray, shape (3, 3)
        Homography mapping source points to destination points.
    """
    return estimate_homography(src_points, dst_points)


def apply_homography(H, points):
    """Apply a homography to 2D points.

    Parameters
    ----------
    H : array-like, shape (3, 3)
        Homography matrix.
    points : array-like, shape (..., 2)
        Source points.

    Returns
    -------
    warped_points : ndarray, shape (..., 2)
        Destination points after homogeneous division.
    """
    pts = np.asarray(points, dtype=float)
    flat = pts.reshape(-1, 2)
    warped = to_homogeneous(flat) @ np.asarray(H, dtype=float).T
    return from_homogeneous(warped).reshape(pts.shape)


def warp_image(image, H, output_shape=None, interpolation="bilinear", fill_value=0):
    """Warp an image with backward mapping.

    Parameters
    ----------
    image : array-like or PIL.Image
        Source grayscale or RGB image.
    H : array-like, shape (3, 3)
        Homography mapping source coordinates to destination coordinates.
    output_shape : tuple[int, int] or None
        Destination image shape as (height, width). If None, use source shape.
    interpolation : {"nearest", "bilinear"}
        Sampling method for non-integer source locations.
    fill_value : int or float
        Value used for destination pixels whose source is outside the image.

    Returns
    -------
    warped : ndarray
        Warped image with shape (height, width) or (height, width, channels).
    """
    img = _as_image_array(image)
    h, w = img.shape[:2]
    out_h, out_w = output_shape or (h, w)
    yy, xx = np.indices((out_h, out_w), dtype=float)
    dst = np.stack([xx.ravel(), yy.ravel()], axis=1)
    src = apply_homography(np.linalg.inv(H), dst)
    warped = _sample_nearest(img, src, fill_value)
    return warped.reshape((out_h, out_w, *img.shape[2:])).astype(img.dtype)


def rectify_plane(image, corners, output_size=None):
    """Rectify a planar quadrilateral into a front-facing rectangle.

    Parameters
    ----------
    image : array-like or PIL.Image
        Source image.
    corners : array-like, shape (4, 2)
        Source corners ordered clockwise or counter-clockwise.
    output_size : tuple[int, int] or None
        Rectified size as (width, height). If None, infer it from side lengths.

    Returns
    -------
    rectified : ndarray
        Rectified image.
    H : ndarray, shape (3, 3)
        Homography mapping source corners to rectangle coordinates.
    """
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (4, 2):
        raise ValueError("corners must have shape (4, 2)")
    if output_size is None:
        top = np.linalg.norm(corners[1] - corners[0])
        bottom = np.linalg.norm(corners[2] - corners[3])
        left = np.linalg.norm(corners[3] - corners[0])
        right = np.linalg.norm(corners[2] - corners[1])
        width = max(1, int(round(max(top, bottom))))
        height = max(1, int(round(max(left, right))))
    else:
        width, height = output_size
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    H = estimate_homography(corners, dst)
    return warp_image(image, H, (height, width)), H


def rectification_target(aspect_ratio, max_output_side=700):
    """Return target rectangle corners and image shape for a known aspect ratio."""
    aspect_ratio = float(aspect_ratio)
    max_output_side = int(max_output_side)
    if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be finite and positive")
    if max_output_side < 2:
        raise ValueError("max_output_side must be at least 2")

    if aspect_ratio >= 1:
        width = max_output_side
        height = max(2, int(round(max_output_side / aspect_ratio)))
    else:
        height = max_output_side
        width = max(2, int(round(max_output_side * aspect_ratio)))
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=float,
    )
    return corners, (height, width)


def load_rectification_photo_set(
    photo_dir,
    object_width=None,
    object_height=None,
    filenames=None,
    max_side=1200,
    minimum_side=512,
):
    """Load a mild/strong planar photo set or construct synthetic demo data."""
    photo_dir = Path(photo_dir)
    filenames = filenames or {"mild": "mild.jpg", "strong": "strong.jpg"}
    paths = {name: photo_dir / filename for name, filename in filenames.items()}
    missing_files = [path.name for path in paths.values() if not path.exists()]
    using_demo_data = bool(missing_files)

    if not using_demo_data:
        photos = {}
        original_sizes = {}
        for name, path in paths.items():
            with Image.open(path) as opened:
                opened = ImageOps.exif_transpose(opened).convert("RGB")
                original_sizes[name] = opened.size
                resampling = getattr(Image, "Resampling", Image)
                opened.thumbnail((max_side, max_side), resampling.LANCZOS)
                photos[name] = np.asarray(opened)

        if object_width is None or object_height is None:
            raise ValueError("Provide the measured object width and height")
        object_width = float(object_width)
        object_height = float(object_height)
        if (
            not np.isfinite(object_width)
            or not np.isfinite(object_height)
            or object_width <= 0
            or object_height <= 0
        ):
            raise ValueError("The measured object dimensions must be finite and positive")
        aspect_ratio = object_width / object_height
        undersized = [
            name
            for name, (width, height) in original_sizes.items()
            if min(width, height) < minimum_side
        ]
    else:
        plane_size = 512
        y, x = np.indices((plane_size, plane_size))
        board = ((x // 64 + y // 64) % 2) > 0
        gray = np.where(board, 220, 55).astype(np.uint8)
        plane = np.repeat(gray[:, :, None], 3, axis=2)
        plane[:10, :] = (245, 180, 30)
        plane[-10:, :] = (245, 180, 30)
        plane[:, :10] = (245, 180, 30)
        plane[:, -10:] = (245, 180, 30)
        plane[135:143, 135:377] = (25, 210, 225)
        plane[369:377, 135:377] = (25, 210, 225)
        plane[135:377, 135:143] = (25, 210, 225)
        plane[135:377, 369:377] = (25, 210, 225)
        source_corners = np.array(
            [[0, 0], [511, 0], [511, 511], [0, 511]], dtype=float
        )
        quadrilaterals = {
            "mild": np.array(
                [[95, 75], [605, 95], [585, 620], [115, 600]], dtype=float
            ),
            "strong": np.array(
                [[165, 70], [610, 195], [515, 565], [85, 650]], dtype=float
            ),
        }
        photos = {
            name: warp_image(
                plane,
                estimate_homography(source_corners, quadrilateral),
                output_shape=(700, 700),
            )
            for name, quadrilateral in quadrilaterals.items()
        }
        original_sizes = {name: (700, 700) for name in photos}
        aspect_ratio = 1.0
        undersized = []

    return {
        "photos": photos,
        "original_sizes": original_sizes,
        "aspect_ratio": float(aspect_ratio),
        "using_demo_data": using_demo_data,
        "submission_data_pass": not using_demo_data and not undersized,
        "missing_files": missing_files,
        "undersized": undersized,
    }


# Backward-compatible alias for early drafts of the homework helper.
load_rectification_photo_pair = load_rectification_photo_set


def rectification_baseline_results(comparison_state):
    """Validate a comparison state and return held-out baseline measurements."""
    incomplete = []
    for view_name, view in comparison_state["views"].items():
        for point_set in ("outer", "inner"):
            if len(view[point_set]) != 4:
                incomplete.append(f"{view_name}/{point_set}: {len(view[point_set])}/4")
    if incomplete:
        raise RuntimeError(
            "Complete these point sets in the interactive cell: " + ", ".join(incomplete)
        )

    rows = []
    for view_name, view in comparison_state["views"].items():
        if view.get("metrics") is None:
            raise RuntimeError(f"No rectification metrics are available for {view_name}")
        for model in ("affine", "homography"):
            rows.append(
                {
                    "view": view_name,
                    "model": model,
                    **view["metrics"][model],
                }
            )
    return rows


def jitter_distribution_diagnostics(
    points,
    jitter_function,
    sigma_px=8,
    trials=400,
    seed=320,
):
    """Exercise a student jitter function and summarize its sampled offsets."""
    points = np.asarray(points, dtype=float)
    sigma_px = float(sigma_px)
    trials = int(trials)
    if points.shape != (4, 2):
        raise ValueError("points must contain four ordered 2D corners")
    if not np.isfinite(sigma_px) or sigma_px <= 0:
        raise ValueError("Diagnostic sigma_px must be finite and positive")
    if trials < 2:
        raise ValueError("trials must be at least 2")

    points_before = points.copy()
    rng = np.random.default_rng(seed)
    samples = np.stack(
        [jitter_function(points, sigma_px, rng) for _ in range(trials)]
    )
    expected_shape = (trials, *points.shape)
    if samples.shape != expected_shape:
        raise ValueError(
            f"jitter_points produced shape {samples.shape}; expected {expected_shape}"
        )
    if not np.array_equal(points, points_before):
        raise ValueError("jitter_points modified its input array")
    zero_jitter = jitter_function(points, 0, np.random.default_rng(0))
    if not np.array_equal(zero_jitter, points):
        raise ValueError("sigma_px=0 must return coordinates equal to the input")
    for invalid_sigma in (-1, np.nan, np.inf):
        try:
            jitter_function(points, invalid_sigma, np.random.default_rng(0))
        except ValueError:
            pass
        else:
            raise ValueError(
                "Negative and non-finite sigma_px values must raise ValueError"
            )

    offsets = samples - points[None, :, :]
    flat_offsets = offsets.reshape(-1, 2)
    coordinate_offsets = offsets.reshape(trials, -1)
    empirical_mean = flat_offsets.mean(axis=0)
    empirical_std = flat_offsets.std(axis=0, ddof=1)
    correlation = np.corrcoef(coordinate_offsets, rowvar=False)
    largest_cross_correlation = float(
        np.max(np.abs(correlation - np.eye(correlation.shape[0])))
    )
    plausible = bool(
        np.all(np.abs(empirical_mean) <= 0.25 * sigma_px)
        and np.all(np.abs(empirical_std - sigma_px) <= 0.25 * sigma_px)
        and largest_cross_correlation <= 0.25
    )
    return {
        "points": points,
        "samples": samples,
        "offsets": offsets,
        "sigma_px": sigma_px,
        "trials": trials,
        "seed": seed,
        "empirical_mean": empirical_mean,
        "empirical_std": empirical_std,
        "largest_cross_correlation": largest_cross_correlation,
        "plausible": plausible,
    }


def rectification_jitter_experiment(
    comparison_state,
    jitter_function,
    noise_levels=(0, 2, 5, 10),
    trials=40,
    seed=320,
    view_names=None,
):
    """Refit affine and projective rectifications under simulated click noise."""
    rectification_baseline_results(comparison_state)
    noise_levels = tuple(float(value) for value in noise_levels)
    if not noise_levels or any(not np.isfinite(value) or value < 0 for value in noise_levels):
        raise ValueError("noise_levels must contain finite nonnegative values")
    trials = int(trials)
    if trials < 1:
        raise ValueError("trials must be positive")
    if view_names is None:
        view_names = tuple(comparison_state["views"])
    else:
        view_names = tuple(view_names)
    unknown_views = [name for name in view_names if name not in comparison_state["views"]]
    if unknown_views:
        raise KeyError("Unknown comparison views: " + ", ".join(unknown_views))

    rng = np.random.default_rng(seed)
    trial_rows = []
    for view_name in view_names:
        view = comparison_state["views"][view_name]
        outer = np.asarray(view["outer"], dtype=float)
        inner = np.asarray(view["inner"], dtype=float)
        destination = np.asarray(view["destination_corners"], dtype=float)
        for noise_px in noise_levels:
            for trial in range(trials):
                noisy_outer = jitter_function(outer, noise_px, rng)
                transforms = {
                    "affine": estimate_affine_transform(noisy_outer, destination),
                    "homography": estimate_homography(noisy_outer, destination),
                }
                for model, transform in transforms.items():
                    metrics = rectangle_geometry_metrics(
                        apply_homography(transform, inner)
                    )
                    trial_rows.append(
                        {
                            "view": view_name,
                            "model": model,
                            "noise_px": noise_px,
                            "trial": trial,
                            **metrics,
                        }
                    )

    summary = []
    for view_name in view_names:
        for model in ("affine", "homography"):
            for noise_px in noise_levels:
                values = np.array(
                    [
                        row["right_angle_error_deg"]
                        for row in trial_rows
                        if row["view"] == view_name
                        and row["model"] == model
                        and row["noise_px"] == noise_px
                    ]
                )
                summary.append(
                    {
                        "view": view_name,
                        "model": model,
                        "noise_px": noise_px,
                        "median": float(np.median(values)),
                        "q25": float(np.percentile(values, 25)),
                        "q75": float(np.percentile(values, 75)),
                    }
                )
    return {"trials": trial_rows, "summary": summary}


def line_from_points(p1, p2):
    """Return the homogeneous line passing through two image points.

    Parameters
    ----------
    p1, p2 : array-like, shape (2,)
        Two points on the line.

    Returns
    -------
    line : ndarray, shape (3,)
        Homogeneous line coefficients [a, b, c] for ax + by + c = 0.
    """
    line = np.cross(to_homogeneous(p1), to_homogeneous(p2))
    norm = np.linalg.norm(line[:2])
    return line / norm if norm > 0 else line


def intersect_lines(line1, line2):
    """Intersect two homogeneous 2D lines.

    Parameters
    ----------
    line1, line2 : array-like, shape (3,)
        Lines in ax + by + c = 0 form.

    Returns
    -------
    point : ndarray, shape (2,)
        Cartesian intersection point. Values may be very large for near-parallel lines.
    """
    return from_homogeneous(np.cross(line1, line2))


def estimate_vanishing_point(lines_or_segments):
    """Estimate a vanishing point from image lines.

    Parameters
    ----------
    lines_or_segments : array-like
        Either lines with shape (N, 3), or clicked line segments with shape (N, 2, 2).

    Returns
    -------
    vp : ndarray, shape (2,)
        Least-squares vanishing point where the lines meet.
    """
    arr = np.asarray(lines_or_segments, dtype=float)
    if arr.ndim == 3 and arr.shape[1:] == (2, 2):
        lines = np.array([line_from_points(seg[0], seg[1]) for seg in arr])
    elif arr.ndim == 2 and arr.shape[1] == 3:
        lines = arr
    else:
        raise ValueError("input must have shape (N, 3) or (N, 2, 2)")
    _, _, vh = np.linalg.svd(lines)
    return from_homogeneous(vh[-1])


def estimate_horizon(vp1, vp2):
    """Estimate the horizon line from two vanishing points.

    Parameters
    ----------
    vp1, vp2 : array-like, shape (2,)
        Vanishing points for two horizontal world directions.

    Returns
    -------
    horizon : ndarray, shape (3,)
        Homogeneous line coefficients [a, b, c].
    """
    return line_from_points(vp1, vp2)


def camera_project(
    points_3d,
    focal_length=800.0,
    image_center=(320.0, 240.0),
    camera_position=(0.0, 0.0, 0.0),
    rotation_degrees=(0.0, 0.0, 0.0),
):
    """Project 3D points into a pinhole camera.

    Parameters
    ----------
    points_3d : array-like, shape (..., 3)
        World points in camera-compatible units.
    focal_length : float
        Focal length in pixels.
    image_center : tuple[float, float]
        Principal point (cx, cy) in pixels.
    camera_position : tuple[float, float, float]
        Camera center in world coordinates.
    rotation_degrees : tuple[float, float, float]
        Rotations around the camera x, y, and z axes in degrees.

    Returns
    -------
    image_points : ndarray, shape (..., 2)
        Projected pixel coordinates.
    depths : ndarray, shape (...)
        Depth values in the camera coordinate frame.
    """
    pts = np.asarray(points_3d, dtype=float)
    shape = pts.shape[:-1]
    flat = pts.reshape(-1, 3)
    R = _rotation_matrix(*np.deg2rad(rotation_degrees))
    cam = (R @ (flat - np.asarray(camera_position, dtype=float)).T).T
    z = cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-12, np.nan, z)
    cx, cy = image_center
    xy = np.column_stack(
        [focal_length * cam[:, 0] / safe_z + cx, focal_length * cam[:, 1] / safe_z + cy]
    )
    return xy.reshape((*shape, 2)), z.reshape(shape)


def camera_extrinsic_matrices(rotation_degrees=(0.0, 0.0, 0.0), camera_position=(0.0, 0.0, 0.0)):
    """Return translation, rotation, and combined world-to-camera matrices.

    Parameters
    ----------
    rotation_degrees : tuple[float, float, float]
        Rotations around the camera x, y, and z axes in degrees.
    camera_position : tuple[float, float, float]
        Camera center in world coordinates.

    Returns
    -------
    T : ndarray, shape (4, 4)
        Translation matrix moving the world by -camera_position.
    R : ndarray, shape (4, 4)
        Rotation matrix orienting the camera.
    E : ndarray, shape (4, 4)
        Combined extrinsic matrix E = R @ T.
    """
    R3 = _rotation_matrix(*np.deg2rad(rotation_degrees))
    T = np.eye(4)
    T[:3, 3] = -np.asarray(camera_position, dtype=float)
    R = np.eye(4)
    R[:3, :3] = R3
    return T, R, R @ T


def _normalize_points(points):
    center = points.mean(axis=0)
    shifted = points - center
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0
    T = np.array([[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]], [0, 0, 1]])
    return apply_homography(T, points), T


def _as_image_array(image):
    if isinstance(image, Image.Image):
        return np.asarray(image)
    return np.asarray(image)


def _sample_nearest(img, xy, fill_value):
    h, w = img.shape[:2]
    x = np.rint(xy[:, 0]).astype(int)
    y = np.rint(xy[:, 1]).astype(int)
    valid = (0 <= x) & (x < w) & (0 <= y) & (y < h)
    out = np.full((xy.shape[0], *img.shape[2:]), fill_value, dtype=img.dtype)
    out[valid] = img[y[valid], x[valid]]
    return out


def _rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx
