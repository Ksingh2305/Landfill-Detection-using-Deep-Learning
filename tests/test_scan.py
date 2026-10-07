"""Tiling geometry, detection merging and overlay rendering."""

from __future__ import annotations

import numpy as np
import pytest

from src.scan import (
    Detection,
    _iou,
    draw_detections,
    extract_tiles,
    merge_detections,
    tile_positions,
)


# ----------------------------------------------------------------------
# Tiling
# ----------------------------------------------------------------------
def test_tile_positions_cover_the_whole_image():
    h, w, tile, stride = 300, 500, 128, 64
    positions = tile_positions(h, w, tile, stride)
    covered = np.zeros((h, w), dtype=bool)
    for y, x in positions:
        covered[y : y + tile, x : x + tile] = True
    assert covered.all(), "sliding window left uncovered pixels"


def test_tile_positions_never_run_off_the_edge():
    h, w, tile = 200, 275, 96
    for y, x in tile_positions(h, w, tile, 48):
        assert 0 <= y and y + tile <= h
        assert 0 <= x and x + tile <= w


def test_overlap_increases_tile_count():
    few = len(tile_positions(512, 512, 128, 128))
    many = len(tile_positions(512, 512, 128, 64))
    assert many > few


def test_extract_tiles_shapes():
    image = np.random.randint(0, 255, (256, 320, 3), dtype=np.uint8)
    tiles, positions = extract_tiles(image, 128, 64)
    assert tiles.shape[1:] == (128, 128, 3)
    assert len(tiles) == len(positions)


def test_extract_tiles_respects_the_safety_cap():
    image = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="cap"):
        extract_tiles(image, 64, 8, max_tiles=100)


def test_tile_larger_than_image_is_clamped():
    image = np.random.randint(0, 255, (40, 50, 3), dtype=np.uint8)
    tiles, _ = extract_tiles(image, 128, 64)
    assert tiles.shape[1] <= 40 and tiles.shape[2] <= 40


# ----------------------------------------------------------------------
# Merging
# ----------------------------------------------------------------------
def test_iou_basics():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0 < _iou((0, 0, 10, 10), (5, 5, 15, 15)) < 1


def test_merge_collapses_overlapping_windows():
    # b0 and b1 overlap at IoU 0.32; the grown box then swallows b2 at IoU 0.22.
    boxes = [(0, 0, 100, 100), (30, 30, 130, 130), (60, 60, 160, 160)]
    dets = merge_detections(boxes, [0.9, 0.8, 0.7], iou_threshold=0.15)
    assert len(dets) == 1
    d = dets[0]
    assert (d.x, d.y) == (0, 0)
    assert d.width == 160 and d.height == 160
    assert d.n_tiles == 3
    assert d.confidence == pytest.approx(0.9)


def test_merge_keeps_distant_windows_separate():
    boxes = [(0, 0, 50, 50), (500, 500, 550, 550)]
    dets = merge_detections(boxes, [0.9, 0.85], iou_threshold=0.3)
    assert len(dets) == 2
    assert dets[0].confidence >= dets[1].confidence


def test_merge_handles_empty_input():
    assert merge_detections([], []) == []


def test_detection_area_conversion():
    det = Detection(x=0, y=0, width=100, height=50, confidence=0.9,
                    mean_confidence=0.8, n_tiles=2)
    assert det.area_pixels() == 5000
    assert det.area_m2(10.0) == pytest.approx(500_000.0)
    d = det.as_dict(metres_per_pixel=10.0)
    assert d["area_hectares"] == pytest.approx(50.0)


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def test_draw_detections_marks_pixels_without_changing_shape():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    det = Detection(x=20, y=30, width=60, height=40, confidence=0.9,
                    mean_confidence=0.9, n_tiles=1)
    out = draw_detections(image, [det], thickness=2)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert out.sum() > 0
    assert out[31, 40, 0] > 0, "top edge should be drawn"


def test_draw_detections_clips_to_the_canvas():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    det = Detection(x=-20, y=-20, width=200, height=200, confidence=0.5,
                    mean_confidence=0.5, n_tiles=1)
    out = draw_detections(image, [det])
    assert out.shape == image.shape


# ----------------------------------------------------------------------
# End to end with a real (untrained) model
# ----------------------------------------------------------------------
def test_scan_image_end_to_end(tmp_config):
    pytest.importorskip("tensorflow")
    from src.data.synthetic import make_scene
    from src.models.build import build_model
    from src.scan import render_overlay, scan_image

    scene, _ = make_scene(320, 256, seed=5)
    model = build_model(tmp_config)

    result = scan_image(scene, tmp_config, model=model,
                        meta={"image_size": list(tmp_config.image_size)},
                        tile_size=64, overlap=0.5, threshold=0.5)

    assert result.probability_map.shape == scene.shape[:2]
    assert result.probability_map.min() >= 0.0
    assert result.probability_map.max() <= 1.0
    assert result.n_tiles == len(result.tile_boxes)

    summary = result.summary(metres_per_pixel=10.0)
    assert summary["n_tiles"] > 0
    assert summary["scene_shape"] == [256, 320]

    overlay = render_overlay(scene, result)
    assert overlay.shape == scene.shape
    assert overlay.dtype == np.uint8
