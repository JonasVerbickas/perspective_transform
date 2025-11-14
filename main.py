"""Interactive perspective transform demo built with OpenCV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:  # Optional convenience picker for image selection.
    from tkinter import Tk, filedialog
except Exception:  # pragma: no cover - GUI helper only.
    Tk = None
    filedialog = None


WINDOW_NAME = "Perspective Transform Demo"
POINT_COLOR = (0, 255, 0)
SELECTED_COLOR = (0, 165, 255)
LINE_COLOR = (255, 255, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GUI demo that lets you drag perspective source points."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to the input image. If omitted, a file dialog opens (when available).",
    )
    return parser.parse_args()


def pick_image_via_dialog() -> Optional[str]:
    if Tk is None or filedialog is None:
        return None
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an input image",
        filetypes=[
            ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return file_path or None


def load_image(path_str: str) -> np.ndarray:
    image = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image at '{path_str}'.")
    return image


def add_padding(image: np.ndarray, pad_ratio: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    pad = max(1, int(round(pad_ratio * max(height, width))))
    padded = cv2.copyMakeBorder(
        image,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    initial_points = np.array(
        [
            [pad, pad],
            [pad + width - 1, pad],
            [pad + width - 1, pad + height - 1],
            [pad, pad + height - 1],
        ],
        dtype=np.float32,
    )
    return padded, initial_points


def _order_points(points: np.ndarray) -> np.ndarray:
    """Concise method of ordering corners of 4 vertex polygons.

    Sum:
    Top-left -> coord pair sum is minimized.
    Bottom-right -> coord pair sum is maximized.

    Diff (can be negative -> minimized when x>>y):
    Top-right -- coord pair difference is minimized -> y<x.
    Bottom-left -- coord pair difference is maximized -> y>x.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    rect[0] = points[np.argmin(s)]  # top-left
    rect[2] = points[np.argmax(s)]  # bottom-right
    rect[1] = points[np.argmin(diff)]  # top-right
    rect[3] = points[np.argmax(diff)]  # bottom-left
    return rect

def order_points(points: np.ndarray) -> np.ndarray:
    """Return a new array where points are ordered [tl, tr, br, bl]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    points_in_order_of_ascending_y_coord = np.argsort(points[:, 1])
    # TOP
    topmost_point_pair = points_in_order_of_ascending_y_coord[:2]
    if points[topmost_point_pair[0]][0] < points[topmost_point_pair[1]][0]:
        topmost_pair_lower_x_index = topmost_point_pair[0]
        topmost_pair_higher_x_index = topmost_point_pair[1]
    else:
        topmost_pair_lower_x_index = topmost_point_pair[1]
        topmost_pair_higher_x_index = topmost_point_pair[0]
    rect[0] = points[topmost_pair_lower_x_index]  # top-left
    rect[1] = points[topmost_pair_higher_x_index]  # top-right
    # BOTTOM
    bottommost_point_pair = points_in_order_of_ascending_y_coord[-2:]
    if points[bottommost_point_pair[0]][0] < points[bottommost_point_pair[1]][0]:
        bottommost_pair_lower_x_index = bottommost_point_pair[0]
        bottommost_pair_higher_x_index = bottommost_point_pair[1]
    else:
        bottommost_pair_lower_x_index = bottommost_point_pair[1]
        bottommost_pair_higher_x_index = bottommost_point_pair[0]
    rect[2] = points[bottommost_pair_higher_x_index]  # bottom-right
    rect[3] = points[bottommost_pair_lower_x_index]  # bottom-left
    return rect


def warp_from_points(
    image: np.ndarray,
    points: np.ndarray,
    canvas_size: tuple[int, int] | None = None,
) -> np.ndarray:
    rect = order_points(points)
    width_a = float(np.linalg.norm(rect[2] - rect[3]))
    width_b = float(np.linalg.norm(rect[1] - rect[0]))
    height_a = float(np.linalg.norm(rect[1] - rect[2]))
    height_b = float(np.linalg.norm(rect[0] - rect[3]))
    target_w = max(1, int(round(max(width_a, width_b))))
    target_h = max(1, int(round(max(height_a, height_b))))
    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (target_w, target_h))

    if not canvas_size:
        return warped

    canvas_w, canvas_h = canvas_size
    if warped.shape[1] == canvas_w and warped.shape[0] == canvas_h:
        return warped

    scale = min(canvas_w / max(1, warped.shape[1]), canvas_h / max(1, warped.shape[0]))
    scaled_w = max(1, int(round(warped.shape[1] * scale)))
    scaled_h = max(1, int(round(warped.shape[0] * scale)))
    if (scaled_w, scaled_h) != (warped.shape[1], warped.shape[0]):
        interpolation = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
        warped = cv2.resize(warped, (scaled_w, scaled_h), interpolation=interpolation)

    result = np.zeros((canvas_h, canvas_w, 3), dtype=warped.dtype)
    offset_x = (canvas_w - warped.shape[1]) // 2
    offset_y = (canvas_h - warped.shape[0]) // 2
    result[offset_y : offset_y + warped.shape[0], offset_x : offset_x + warped.shape[1]] = warped
    return result


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    scale = height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def draw_overlay(image: np.ndarray, points: np.ndarray, selected: Optional[int]) -> np.ndarray:
    overlay = image.copy()
    pts_int = points.astype(int)
    cv2.polylines(overlay, [pts_int], isClosed=True, color=LINE_COLOR, thickness=2)
    for idx, (x, y) in enumerate(pts_int):
        color = SELECTED_COLOR if idx == selected else POINT_COLOR
        cv2.circle(overlay, (x, y), 8, color, thickness=-1)
    cv2.putText(
        overlay,
        "Drag points | R: reset | Esc/Q: quit",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def hit_test(points: np.ndarray, x: int, y: int, radius: int = 15) -> Optional[int]:
    distances = np.linalg.norm(points - np.array([x, y], dtype=np.float32), axis=1)
    idx = int(np.argmin(distances))
    if distances[idx] <= radius:
        return idx
    return None


def make_mouse_handler(context: dict):
    def handle(event: int, x: int, y: int, flags: int, _param):
        left_w, left_h = context["left_size"]
        points = context["points"]
        if x >= left_w or y >= left_h:
            if event == cv2.EVENT_LBUTTONUP:
                context["selected"] = None
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            hit = hit_test(points, x, y)
            if hit is not None:
                context["selected"] = hit
        elif event == cv2.EVENT_LBUTTONUP:
            context["selected"] = None
        elif event == cv2.EVENT_MOUSEMOVE and context["selected"] is not None:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                idx = context["selected"]
                points[idx] = [
                    float(np.clip(x, 0, left_w - 1)),
                    float(np.clip(y, 0, left_h - 1)),
                ]

    return handle


def build_display(left: np.ndarray, warped: np.ndarray) -> np.ndarray:
    right = resize_to_height(warped, left.shape[0])
    return np.hstack((left, right))


def main() -> None:
    args = parse_args()
    path_str = args.image or pick_image_via_dialog()
    if not path_str:
        raise SystemExit("No image selected. Provide a path or select a file when prompted.")

    image_path = Path(path_str)
    if not image_path.exists():
        raise SystemExit(f"Image '{image_path}' does not exist.")

    raw = load_image(str(image_path))
    padded, initial_points = add_padding(raw)
    points = initial_points.copy()

    context = {
        "points": points,
        "selected": None,
        "left_size": (padded.shape[1], padded.shape[0]),
        "initial": initial_points,
    }

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, make_mouse_handler(context))

    try:
        while True:
            annotated = draw_overlay(padded, points, context["selected"])
            try:
                warped = warp_from_points(
                    padded,
                    points,
                    canvas_size=(padded.shape[1], padded.shape[0]),
                )
            except cv2.error:
                warped = np.zeros_like(padded)
            display = build_display(annotated, warped)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(16) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                points[:] = context["initial"]
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
