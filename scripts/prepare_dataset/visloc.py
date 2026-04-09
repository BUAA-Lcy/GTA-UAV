# ---------------------------------------------------------------
# Copyright (c) 2024-2025 Yuxiang Ji. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import argparse
import os
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import random
import shutil
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import concurrent.futures
from PIL import Image
import math
from multiprocessing import Pool, cpu_count
import csv
import pickle
import random
import itertools
import json


Image.MAX_IMAGE_PIXELS = None
FOV_V = 52
FOV_H = 36

ALL_SCENE_IDS = list(range(1, 12))
DEFAULT_SAME_AREA_SCENES = [3, 4]
DEFAULT_CROSS_AREA_TRAIN_SCENES = [3]
DEFAULT_CROSS_AREA_TEST_SCENES = [4]

TRAIN_LIST = [1, 3]
TEST_LIST = [2, 4]

# TRAIN_LIST = [1, 2, 3, 4, 5, 8, 11]
# TEST_LIST = [1, 2, 3, 4, 5, 8, 11]

TILE_SIZE = 256

THRESHOLD = 0.39
SEMI_THRESHOLD = 0.14

## Lat, Lon
SATE_LATLON = {
    '01': [29.774065,115.970635,29.702283,115.996851],
    '02': [29.817376,116.033769,29.725402,116.064566],
    '03': [32.355491,119.805926,32.29029,119.900052],
    '04': [32.254036,119.90598,32.151018,119.954509],
    '05': [24.666899,102.340055,24.650422,102.365252],
    '06': [32.373177,109.63516,32.346944,109.656837],
    '07': [40.340058,115.791182,40.339604,115.79923],
    '08': [30.947227,120.136489,30.903521,120.252951],
    '10': [40.355093,115.776356,40.341475,115.794041],
    '11': [38.852301,101.013109,38.807825,101.092483],
}
## H, W
SATE_SIZE = {
    '01': (26762,  9774),
    '02': (34291, 11482),
    '03': (24308, 35092),
    '04': (38408, 18093),
    '05': (6144,   9394),
    '06': (9780,   8082),
    '07': (170,    3000),
    '08': (16294, 43421),
    '10': (5077,   6593),
	'11': (16582, 29592),
}


def format_scene_id(scene_id):
    return f"{int(scene_id):02d}"


def normalize_scene_ids(scene_ids):
    return sorted({int(scene_id) for scene_id in scene_ids})


def parse_scene_ids_arg(scene_ids_text):
    if scene_ids_text is None:
        return None
    scene_ids_text = str(scene_ids_text).strip()
    if not scene_ids_text:
        return None
    if scene_ids_text.lower() in {"all", "*"}:
        return list(ALL_SCENE_IDS)

    scene_ids = []
    for token in scene_ids_text.split(","):
        token = token.strip()
        if not token:
            continue
        scene_ids.append(int(token))
    return normalize_scene_ids(scene_ids)


def get_selected_scene_ids(scene_ids=None):
    if scene_ids is not None:
        return [format_scene_id(scene_id) for scene_id in normalize_scene_ids(scene_ids)]
    return [format_scene_id(scene_id) for scene_id in normalize_scene_ids(TRAIN_LIST + TEST_LIST)]


def configure_scene_lists(split_type, scene_ids=None, train_scene_ids=None, test_scene_ids=None):
    if split_type == 'same-area':
        selected_scene_ids = normalize_scene_ids(scene_ids or DEFAULT_SAME_AREA_SCENES)
        return selected_scene_ids, selected_scene_ids
    if split_type == 'cross-area':
        train_scene_ids = normalize_scene_ids(train_scene_ids or DEFAULT_CROSS_AREA_TRAIN_SCENES)
        test_scene_ids = normalize_scene_ids(test_scene_ids or DEFAULT_CROSS_AREA_TEST_SCENES)
        return train_scene_ids, test_scene_ids
    raise ValueError(f"Unsupported split_type: {split_type}")


def set_scene_globals(train_scene_ids, test_scene_ids):
    global TRAIN_LIST
    global TEST_LIST
    TRAIN_LIST = list(train_scene_ids)
    TEST_LIST = list(test_scene_ids)


def ensure_scene_files_exist(root_dir, scene_ids, require_tile=False):
    missing = []
    for scene_id in [format_scene_id(scene_id) for scene_id in normalize_scene_ids(scene_ids)]:
        scene_dir = os.path.join(root_dir, scene_id)
        csv_path = os.path.join(scene_dir, f'{scene_id}.csv')
        sate_tif_path = os.path.join(scene_dir, f'satellite{scene_id}.tif')
        if not os.path.isfile(csv_path):
            missing.append((scene_id, 'csv', csv_path))
        if not os.path.isfile(sate_tif_path):
            missing.append((scene_id, 'satellite_tif', sate_tif_path))
        if require_tile:
            tile_dir = os.path.join(scene_dir, 'tile')
            if not os.path.isdir(tile_dir):
                missing.append((scene_id, 'tile_dir', tile_dir))
    if missing:
        detail = ', '.join([f'{scene}:{kind}' for scene, kind, _ in missing[:8]])
        raise FileNotFoundError(f"Missing required scene assets under {root_dir}: {detail}")


def compute_same_area_split_index(sample_count, train_ratio):
    if sample_count <= 0:
        return 0
    if sample_count == 1:
        return 1
    train_count = int(round(sample_count * float(train_ratio)))
    train_count = max(1, min(sample_count - 1, train_count))
    return train_count


def process_entries_in_parallel(entries, num_workers=None, desc="Processing"):
    if len(entries) == 0:
        return []

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(process_per_image, entries), total=len(entries), desc=desc):
            results.append(result)
    return [result for result in results if result is not None]


def count_pos_queries(pairs_drone2sate_list):
    return sum(1 for item in pairs_drone2sate_list if len(item.get("pair_pos_sate_img_list", [])) > 0)


def build_split_summary(
        split_type,
        output_prefix,
        train_scene_ids,
        test_scene_ids,
        selected_scene_ids,
        skipped_scenes,
        same_area_split_mode,
        train_ratio,
        split_seed,
        raw_scene_counts,
        processed_data_total,
        processed_data_train,
        processed_data_test,
        output_root,
        save_root,
    ):
    def _scene_stats_from_pairs(pairs):
        stats = {}
        for item in pairs:
            scene_id = item["str_i"]
            stats.setdefault(scene_id, {
                "queries": 0,
                "pos_queries": 0,
                "mean_pos_tiles_per_query": 0.0,
                "mean_semipos_tiles_per_query": 0.0,
            })
            scene_stat = stats[scene_id]
            scene_stat["queries"] += 1
            if len(item.get("pair_pos_sate_img_list", [])) > 0:
                scene_stat["pos_queries"] += 1
            scene_stat["mean_pos_tiles_per_query"] += float(len(item.get("pair_pos_sate_img_list", [])))
            scene_stat["mean_semipos_tiles_per_query"] += float(len(item.get("pair_pos_semipos_sate_img_list", [])))

        for scene_stat in stats.values():
            query_count = max(int(scene_stat["queries"]), 1)
            scene_stat["mean_pos_tiles_per_query"] /= float(query_count)
            scene_stat["mean_semipos_tiles_per_query"] /= float(query_count)
        return stats

    total_stats = _scene_stats_from_pairs(processed_data_total)
    train_stats = _scene_stats_from_pairs(processed_data_train)
    test_stats = _scene_stats_from_pairs(processed_data_test)

    scene_summary = {}
    for scene_id in [format_scene_id(scene_id) for scene_id in normalize_scene_ids(selected_scene_ids)]:
        total_scene_stat = total_stats.get(scene_id, {})
        train_scene_stat = train_stats.get(scene_id, {})
        test_scene_stat = test_stats.get(scene_id, {})
        scene_summary[scene_id] = {
            "raw_queries": int(raw_scene_counts.get(scene_id, 0)),
            "valid_semipos_queries": int(total_scene_stat.get("queries", 0)),
            "valid_pos_queries": int(total_scene_stat.get("pos_queries", 0)),
            "train_queries": int(train_scene_stat.get("queries", 0)),
            "train_pos_queries": int(train_scene_stat.get("pos_queries", 0)),
            "test_queries": int(test_scene_stat.get("queries", 0)),
            "test_pos_queries": int(test_scene_stat.get("pos_queries", 0)),
            "mean_pos_tiles_per_query": float(total_scene_stat.get("mean_pos_tiles_per_query", 0.0)),
            "mean_semipos_tiles_per_query": float(total_scene_stat.get("mean_semipos_tiles_per_query", 0.0)),
            "skipped": scene_id in skipped_scenes,
            "skip_reason": skipped_scenes.get(scene_id),
        }

    return {
        "split_type": split_type,
        "output_prefix": output_prefix,
        "same_area_split_mode": same_area_split_mode,
        "train_ratio": float(train_ratio),
        "split_seed": int(split_seed),
        "train_scene_ids": [format_scene_id(scene_id) for scene_id in normalize_scene_ids(train_scene_ids)],
        "test_scene_ids": [format_scene_id(scene_id) for scene_id in normalize_scene_ids(test_scene_ids)],
        "selected_scene_ids": [format_scene_id(scene_id) for scene_id in normalize_scene_ids(selected_scene_ids)],
        "skipped_scenes": skipped_scenes,
        "overall": {
            "raw_queries": int(sum(raw_scene_counts.get(format_scene_id(scene_id), 0) for scene_id in normalize_scene_ids(selected_scene_ids))),
            "valid_semipos_queries": int(len(processed_data_total)),
            "valid_pos_queries": int(count_pos_queries(processed_data_total)),
            "train_queries": int(len(processed_data_train)),
            "train_pos_queries": int(count_pos_queries(processed_data_train)),
            "test_queries": int(len(processed_data_test)),
            "test_pos_queries": int(count_pos_queries(processed_data_test)),
        },
        "per_scene": scene_summary,
        "generated_files": {
            "train_json": os.path.join(output_root, f'{output_prefix}-drone2sate-train.json'),
            "test_json": os.path.join(output_root, f'{output_prefix}-drone2sate-test.json'),
            "summary_json": os.path.join(output_root, f'{output_prefix}-split-summary.json'),
            "train_pair_meta": os.path.join(save_root, 'train_pair_meta.pkl'),
            "test_pair_meta": os.path.join(save_root, 'test_pair_meta.pkl'),
        },
    }

def tile_center_latlon(left_top_lat, left_top_lon, right_bottom_lat, right_bottom_lon, zoom, x, y, str_i):
    """Calculate the center lat/lon of a tile."""
    sate_h, sate_w = SATE_SIZE[str_i][0], SATE_SIZE[str_i][1]
    max_dim = max(sate_h, sate_w)
    max_zoom = math.ceil(math.log(max_dim / TILE_SIZE, 2))
    scale = 2 ** (max_zoom - zoom)

    scaled_width = math.ceil(sate_w / scale)
    scaled_height = math.ceil(sate_h / scale)

    coe_lon = (x + 0.5) * TILE_SIZE / scaled_width
    coe_lat = (y + 0.5) * TILE_SIZE / scaled_height

    # Calculate the size of each tile in degrees

    lat_diff = left_top_lat - right_bottom_lat
    lon_diff = right_bottom_lon - left_top_lon

    # Calculate the center of the tile in degrees
    center_lat = left_top_lat - coe_lat * lat_diff
    center_lon = left_top_lon + coe_lon * lon_diff

    return center_lat, center_lon

def tile2sate(tile_name):
    tile_name = tile_name.replace('.png', '')
    str_i, zoom_level, tile_x, tile_y = tile_name.split('_')
    zoom_level = int(zoom_level)
    tile_x = int(tile_x)
    tile_y = int(tile_y)
    lt_lat, lt_lon, rb_lat, rb_lon = SATE_LATLON[str_i]
    return tile_center_latlon(lt_lat, lt_lon, rb_lat, rb_lon, zoom_level, tile_x, tile_y, str_i)


def order_points(points):
    hull = ConvexHull(points)
    ordered_points = [points[i] for i in hull.vertices]
    return ordered_points


def calc_intersect_area(poly1, poly2):
    # 计算交集
    intersection = poly1.intersection(poly2)
    return intersection.area


def process_tile(args):
    scaled_image, str_i, zoom_dir, zoom, x, y, tile_size = args
    box = (x, y, min(x + tile_size, scaled_image.width), min(y + tile_size, scaled_image.height))
    tile = scaled_image.crop(box)
    
    # 创建一个透明背景的图像
    transparent_tile = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    
    # 将裁剪后的瓦片粘贴到透明背景中
    transparent_tile.paste(tile, (0, 0))
    
    # Example: 01_7_014_014.png
    transparent_tile.save(os.path.join(zoom_dir, f'{str_i}_{zoom}_{x // tile_size:03}_{y // tile_size:03}.png'))


def tile_satellite(root_dir, scene_ids=None, skip_existing=False, keep_last_n_zooms=0, max_zoom_stub_only=False):
    
    for str_i in get_selected_scene_ids(scene_ids):
        file_dir = os.path.join(root_dir, str_i)
        tile_dir = os.path.join(file_dir, 'tile')
        os.makedirs(tile_dir, exist_ok=True)

        image_path = os.path.join(file_dir, f'satellite{str_i}.tif')
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f'Missing satellite tif for scene {str_i}: {image_path}')
        image = Image.open(image_path)

        # Tile Size
        tile_size = TILE_SIZE

        # Calculate Max Zoom Level
        max_dim = max(image.width, image.height)
        max_zoom = math.ceil(math.log(max_dim / tile_size, 2))
        zoom_levels = list(range(max_zoom + 1))
        if int(keep_last_n_zooms) > 0:
            zoom_levels = zoom_levels[-int(keep_last_n_zooms):]

        if skip_existing:
            zooms_ready = True
            for zoom_idx, zoom in enumerate(zoom_levels):
                scale = 2 ** (max_zoom - zoom)
                scaled_width = math.ceil(image.width / scale)
                scaled_height = math.ceil(image.height / scale)
                expected_tiles = math.ceil(scaled_width / tile_size) * math.ceil(scaled_height / tile_size)
                zoom_dir = os.path.join(tile_dir, str(zoom))
                if max_zoom_stub_only and zoom_idx == len(zoom_levels) - 1:
                    if not os.path.isdir(zoom_dir):
                        zooms_ready = False
                    continue
                existing_tiles = 0
                if os.path.isdir(zoom_dir):
                    existing_tiles = sum(
                        1
                        for file_name in os.listdir(zoom_dir)
                        if os.path.isfile(os.path.join(zoom_dir, file_name)) and file_name.endswith('.png')
                    )
                if existing_tiles != expected_tiles:
                    zooms_ready = False
                    break
            if zooms_ready:
                print(f'Skip tiling {str_i}: required zoom levels already complete')
                image.close()
                continue


        # Tiling
        for zoom_idx, zoom in enumerate(zoom_levels):
            zoom_dir = os.path.join(tile_dir, str(zoom))
            if not os.path.exists(zoom_dir):
                os.makedirs(zoom_dir)
            if max_zoom_stub_only and zoom_idx == len(zoom_levels) - 1:
                print(f'Keep {str_i} zoom {zoom} as directory stub only')
                continue
            
            scale = 2 ** (max_zoom - zoom)
            scaled_width = math.ceil(image.width / scale)
            scaled_height = math.ceil(image.height / scale)
            
            # resize
            scaled_image = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            # Avoid passing the whole resized satellite image into a large
            # multiprocessing task list, which can trigger OOM and get the
            # process killed on high-resolution maps.
            tile_x_count = math.ceil(scaled_width / tile_size)
            tile_y_count = math.ceil(scaled_height / tile_size)
            total_tiles = tile_x_count * tile_y_count

            for x in tqdm(
                range(0, scaled_width, tile_size),
                total=tile_x_count,
                desc=f'Tiling {str_i} zoom {zoom}',
            ):
                for y in range(0, scaled_height, tile_size):
                    process_tile((scaled_image, str_i, zoom_dir, zoom, x, y, tile_size))

            print(f'Finished {str_i} zoom {zoom}: {total_tiles} tiles')

            scaled_image.close()

    print('Tiling Satellite Done')


def copy_satellite(root_dir, scene_ids=None):
    dst_dir = os.path.join(root_dir, 'satellite')
    os.makedirs(dst_dir, exist_ok=True)

    for str_i in get_selected_scene_ids(scene_ids):
        file_dir = os.path.join(root_dir, str_i)
        tile_dir = os.path.join(file_dir, 'tile')
        if not os.path.isdir(tile_dir):
            raise FileNotFoundError(f'Missing tile directory for scene {str_i}: {tile_dir}')

        zoom_list = os.listdir(tile_dir)
        zoom_list = [int(x) for x in zoom_list]
        zoom_list.sort()
        zoom_max = zoom_list[-1]
        ### Only keep the third to second lowest zoom level
        zoom_list = zoom_list[-3:-1]

        for zoom in zoom_list:
            tile_zoom_dir = os.path.join(tile_dir, str(zoom))
            for file_name in os.listdir(tile_zoom_dir):
                source_path = os.path.join(tile_zoom_dir, file_name)
                dst_path = os.path.join(dst_dir, file_name)
                if os.path.isfile(source_path):
                    if os.path.exists(dst_path):
                        continue
                    shutil.copy(source_path, dst_path)

    print('Copy Satellite Done')

     
def copy_drone(root_dir, scene_ids=None):
    dst_dir = os.path.join(root_dir, 'drone', 'images')
    os.makedirs(dst_dir, exist_ok=True)

    for str_i in get_selected_scene_ids(scene_ids):
        file_dir = os.path.join(root_dir, str_i, 'drone')
        if not os.path.isdir(file_dir):
            raise FileNotFoundError(f'Missing drone image directory for scene {str_i}: {file_dir}')
        for file_name in os.listdir(file_dir):
            source_path = os.path.join(file_dir, file_name)
            dst_path = os.path.join(dst_dir, file_name)
            if os.path.isfile(source_path):
                if os.path.exists(dst_path):
                    continue
                shutil.copy(source_path, dst_path)

    print('Copy Drone Done')


def copy_png_files(src_path, dst_path):
    os.makedirs(dst_path, exist_ok=True)

    for root, dirs, files in os.walk(src_path):
        for file_name in files:
            # 检查文件是否为 .png 文件
            if file_name.endswith('.png'):
                # 构建完整的文件路径
                full_file_name = os.path.join(root, file_name)
                if os.path.isfile(full_file_name):
                    # 复制文件到目标文件夹
                    shutil.copy(full_file_name, dst_path)

    print(f"所有 .png 文件已复制到 {dst_path}")


def geo_to_image_coords(lat, lon, lat1, lon1, lat2, lon2, H, W):
    R = 6378137  # 地球半径（米）

    # 计算中心纬度
    center_lat = (lat1 + lat2) / 2

    # 计算地理范围（米）
    x_range = R * (lon2 - lon1) * math.cos(math.radians(center_lat))
    y_range = R * (lat2 - lat1)

    # 计算目标点相对于左上角的平面坐标偏移（米）
    x_offset = R * (lon - lon1) * math.cos(math.radians((lat1 + lat) / 2))
    y_offset = R * (lat - lat1)

    # 计算图像中的坐标
    x = (x_offset / x_range) * W
    y = (y_offset / y_range) * H

    return int(x), int(y)


def offset_to_latlon(latitude, longitude, dx, dy):
    # Earth radius in meters
    R = 6378137
    dlat = dy / R
    dlon = dx / (R * math.cos(math.pi * latitude / 180))
    
    lat_offset = dlat * 180 / math.pi
    lon_offset = dlon * 180 / math.pi
    
    return latitude + lat_offset, longitude + lon_offset


def calculate_coverage_endpoints(heading_angle, height, cur_lat, cur_lon, fov_horizontal=FOV_H, fov_vertical=FOV_V, debug=False):
    # Convert angles from degrees to radians for trigonometric functions
    heading_angle_rad = math.radians(heading_angle)
    fov_horizontal_rad = math.radians(fov_horizontal)
    fov_vertical_rad = math.radians(fov_vertical)
    
    # Calculate the half lengths of the coverage area on the ground
    half_coverage_length_h = height * math.tan(fov_horizontal_rad / 2)
    half_coverage_length_v = height * math.tan(fov_vertical_rad / 2)
    
    # Adjust heading angle for coordinate system where East is 0 and North is 90
    adjusted_heading_angle_rad = math.radians((90 - heading_angle) % 360)
    
    # Calculate the offsets for the four endpoints
    offset_top_left_x = -half_coverage_length_h * math.cos(adjusted_heading_angle_rad) - half_coverage_length_v * math.sin(adjusted_heading_angle_rad)
    offset_top_left_y = -half_coverage_length_h * math.sin(adjusted_heading_angle_rad) + half_coverage_length_v * math.cos(adjusted_heading_angle_rad)
    
    offset_top_right_x = half_coverage_length_h * math.cos(adjusted_heading_angle_rad) - half_coverage_length_v * math.sin(adjusted_heading_angle_rad)
    offset_top_right_y = half_coverage_length_h * math.sin(adjusted_heading_angle_rad) + half_coverage_length_v * math.cos(adjusted_heading_angle_rad)
    
    offset_bottom_left_x = -half_coverage_length_h * math.cos(adjusted_heading_angle_rad) + half_coverage_length_v * math.sin(adjusted_heading_angle_rad)
    offset_bottom_left_y = -half_coverage_length_h * math.sin(adjusted_heading_angle_rad) - half_coverage_length_v * math.cos(adjusted_heading_angle_rad)
    
    offset_bottom_right_x = half_coverage_length_h * math.cos(adjusted_heading_angle_rad) + half_coverage_length_v * math.sin(adjusted_heading_angle_rad)
    offset_bottom_right_y = half_coverage_length_h * math.sin(adjusted_heading_angle_rad) - half_coverage_length_v * math.cos(adjusted_heading_angle_rad)
    
    if debug:
        print(
            'offset',
            offset_top_left_x, 
            offset_top_left_y,
            offset_top_right_x, 
            offset_top_right_y,
            offset_bottom_left_x, 
            offset_bottom_left_y,
            offset_bottom_right_x, 
            offset_bottom_right_y
        )

    return {
        "top_left": offset_to_latlon(cur_lat, cur_lon, offset_top_left_x, offset_top_left_y),
        "top_right": offset_to_latlon(cur_lat, cur_lon, offset_top_right_x, offset_top_right_y),
        "bottom_left": offset_to_latlon(cur_lat, cur_lon, offset_bottom_left_x, offset_bottom_left_y),
        "bottom_right": offset_to_latlon(cur_lat, cur_lon, offset_bottom_right_x, offset_bottom_right_y)
    }


def tile_expand(str_i, cur_tile_x, cur_tile_y, p_img_xy_scale, zoom_level, tile_x_max, tile_y_max, debug=False):
    tile_area = TILE_SIZE ** 2

    tile_u = max(0, cur_tile_y - 5)
    tile_d = min(cur_tile_y + 5, tile_y_max)
    tile_l = max(0, cur_tile_x - 5)
    tile_r = min(cur_tile_x + 5, tile_x_max)

    p_img_xy_scale_order = order_points(p_img_xy_scale)
    poly_p = Polygon(p_img_xy_scale_order)
    poly_p_area = poly_p.area

    tile_tmp = [((cur_tile_x    ) * TILE_SIZE, (cur_tile_y    ) * TILE_SIZE), 
                ((cur_tile_x + 1) * TILE_SIZE, (cur_tile_y    ) * TILE_SIZE), 
                ((cur_tile_x    ) * TILE_SIZE, (cur_tile_y + 1) * TILE_SIZE), 
                ((cur_tile_x + 1) * TILE_SIZE, (cur_tile_y + 1) * TILE_SIZE)]
    tile_tmp_order = order_points(tile_tmp)
    poly_tile = Polygon(tile_tmp_order)
    poly_tile_area = poly_tile.area

    tile_iou_expand_list = []
    tile_iou_expand_weight_list = []
    tile_iou_expand_loc_lat_lon_list = []
    tile_semi_iou_expand_list = []
    tile_semi_iou_expand_weight_list = []
    tile_semi_iou_expand_loc_lat_lon_list = []

    for tile_x_i in range(tile_l, tile_r + 1):
        for tile_y_i in range(tile_u, tile_d + 1):

            tile_tmp = [((tile_x_i    ) * TILE_SIZE, (tile_y_i    ) * TILE_SIZE), 
                        ((tile_x_i + 1) * TILE_SIZE, (tile_y_i    ) * TILE_SIZE), 
                        ((tile_x_i    ) * TILE_SIZE, (tile_y_i + 1) * TILE_SIZE), 
                        ((tile_x_i + 1) * TILE_SIZE, (tile_y_i + 1) * TILE_SIZE)]
            tile_tmp_order = order_points(tile_tmp)
            poly_tile = Polygon(tile_tmp_order)
            poly_tile_area = poly_tile.area
            intersect_area = calc_intersect_area(poly_p, poly_tile)
            if debug:
                print('zoom=', zoom_level, cur_tile_x, cur_tile_y)
                print(tile_x_i, tile_y_i)
                print(intersect_area, tile_area, poly_p_area, intersect_area/tile_area, intersect_area/poly_p_area)
            oc = intersect_area / min(poly_p_area, poly_tile_area)
            iou = intersect_area / (poly_p_area + poly_tile_area - intersect_area)
            if iou > THRESHOLD:
                tile_name = f'{str_i}_{zoom_level}_{tile_x_i:03}_{tile_y_i:03}.png'
                tile_iou_expand_list.append(tile_name)
                tile_iou_expand_weight_list.append(iou)
                tile_iou_expand_loc_lat_lon_list.append(tile2sate(tile_name))
            if iou > SEMI_THRESHOLD:
                tile_name = f'{str_i}_{zoom_level}_{tile_x_i:03}_{tile_y_i:03}.png'
                tile_semi_iou_expand_list.append(tile_name)
                tile_semi_iou_expand_weight_list.append(iou)
                tile_semi_iou_expand_loc_lat_lon_list.append(tile2sate(tile_name))
            
    return tile_iou_expand_list, tile_iou_expand_weight_list, tile_iou_expand_loc_lat_lon_list, tile_semi_iou_expand_list, tile_semi_iou_expand_weight_list, tile_semi_iou_expand_loc_lat_lon_list

def process_per_image(drone_meta_data):
    file_dir, str_i, drone_img, lat, lon, height, omega, kappa, phi1, phi2, sate_lt_lat, sate_lt_lon, sate_rb_lat, sate_rb_lon, sate_pix_h, sate_pix_w = drone_meta_data

    # debug = (drone_img == '01_0015.JPG')
    debug = False

    p_latlon = calculate_coverage_endpoints(heading_angle=phi1, height=height, cur_lat=lat, cur_lon=lon, debug=debug)
    
    if debug:
        print(p_latlon)

    zoom_list = os.listdir(os.path.join(file_dir, 'tile'))
    zoom_list = [int(x) for x in zoom_list]
    zoom_list.sort()
    zoom_max = zoom_list[-1]
    ### Only keep the third to second lowest zoom level
    zoom_list = zoom_list[-3:-1]

    cur_img_x, cur_img_y = geo_to_image_coords(lat, lon, sate_lt_lat, sate_lt_lon, sate_rb_lat, sate_rb_lon, sate_pix_h, sate_pix_w)
    p_img_xy = [
        geo_to_image_coords(v[0], v[1], sate_lt_lat, sate_lt_lon, sate_rb_lat, sate_rb_lon, sate_pix_h, sate_pix_w)
            for v in p_latlon.values()
    ]
    if debug:
        print('p_img_xy', p_img_xy)

    result = {
        "str_i": str_i,
        "drone_img_dir": os.path.join(file_dir, 'drone'),
        "drone_img": drone_img,
        "lat": lat,
        "lon": lon,
        "height": height,
        "omega": omega,
        "kappa": kappa,
        "phi1": phi1,
        "phi2": phi2,
        "drone_yaw": phi1,
        "cam_yaw": phi1,
        "sate_img_dir": os.path.join(os.path.dirname(file_dir), 'satellite'),
        "pair_pos_sate_img_list": [],
        "pair_pos_sate_weight_list": [],
        "pair_pos_sate_loc_lat_lon_list": [],
        "pair_pos_semipos_sate_img_list": [],
        "pair_pos_semipos_sate_weight_list": [],
        "pair_pos_semipos_sate_loc_lat_lon_list": [],
    }

    for zoom_level in zoom_list:
        scale = 2 ** (zoom_max - zoom_level)
        sate_pix_w_scale = math.ceil(sate_pix_w / scale)
        sate_pix_h_scale = math.ceil(sate_pix_h / scale)
        
        tile_x_max = sate_pix_w_scale // TILE_SIZE
        tile_y_max = sate_pix_h_scale // TILE_SIZE

        cur_img_x_scale = math.ceil(cur_img_x / scale)
        cur_img_y_scale = math.ceil(cur_img_y / scale)

        p_img_xy_scale = [
            (math.ceil(v[0] / scale), math.ceil(v[1] / scale)) 
                for v in p_img_xy
        ]

        cur_tile_x = cur_img_x_scale // TILE_SIZE
        cur_tile_y = cur_img_y_scale // TILE_SIZE

        tile_iou_expand_list, tile_iou_expand_weight_list, tile_iou_expand_loc_lat_lon_list, \
        tile_semi_iou_expand_list, tile_semi_iou_expand_weight_list, tile_semi_iou_expand_loc_lat_lon_list \
            = tile_expand(str_i, cur_tile_x, cur_tile_y, p_img_xy_scale, zoom_level, tile_x_max, tile_y_max, debug)

        result["pair_pos_sate_img_list"].extend(tile_iou_expand_list)
        result["pair_pos_sate_weight_list"].extend(tile_iou_expand_weight_list)
        result["pair_pos_sate_loc_lat_lon_list"].extend(tile_iou_expand_loc_lat_lon_list)
        result["pair_pos_semipos_sate_img_list"].extend(tile_semi_iou_expand_list)
        result["pair_pos_semipos_sate_weight_list"].extend(tile_semi_iou_expand_weight_list)
        result["pair_pos_semipos_sate_loc_lat_lon_list"].extend(tile_semi_iou_expand_loc_lat_lon_list)

    if len(result["pair_pos_semipos_sate_img_list"]) == 0:
        return None

    if debug:
        print(result)
    
    return result


def save_pairs_meta_data(pairs_drone2sate_list, pkl_save_path, pair_save_dir):
    pairs_iou_sate2drone_dict = {}
    pairs_iou_drone2sate_dict = {}
    pairs_semi_iou_sate2drone_dict = {}
    pairs_semi_iou_drone2sate_dict = {}
    
    # drone_save_dir = os.path.join(pair_save_dir, 'drone')
    # sate_iou_save_dir = os.path.join(pair_save_dir, 'satellite', 'iou')
    # sate_semi_iou_save_dir = os.path.join(pair_save_dir, 'satellite', 'semi_iou')
    # os.makedirs(drone_save_dir, exist_ok=True)
    # os.makedirs(sate_iou_save_dir, exist_ok=True)
    # os.makedirs(sate_semi_iou_save_dir, exist_ok=True)

    pairs_drone2sate_list_save = []

    for pairs_drone2sate in pairs_drone2sate_list:
        
        str_i = pairs_drone2sate['str_i']
        pair_pos_sate_img_list = pairs_drone2sate["pair_pos_sate_img_list"]
        pair_pos_semipos_sate_img_list = pairs_drone2sate["pair_pos_semipos_sate_img_list"]

        drone_img = pairs_drone2sate["drone_img"]
        drone_img_dir = pairs_drone2sate["drone_img_dir"]
        drone_img_name = drone_img.replace('.JPG', '')
        sate_img_dir = pairs_drone2sate["sate_img_dir"]

        ## Check if sate_img exist
        flag = False
        for sate_img in pair_pos_sate_img_list:
            if os.path.exists(os.path.join(sate_img_dir, sate_img)):
                # pairs_pos_drone2sate_dict.setdefault(drone_img, []).append(f'{sate_img}')
                # pairs_pos_sate2drone_dict.setdefault(f'{sate_img}', []).append(f'{drone_img}')
                flag = True
        for sate_img in pair_pos_semipos_sate_img_list:
            if os.path.exists(os.path.join(sate_img_dir, sate_img)):
                # pairs_pos_semipos_drone2sate_dict.setdefault(drone_img, []).append(f'{sate_img}')
                # pairs_pos_semipos_sate2drone_dict.setdefault(f'{sate_img}', []).append(f'{drone_img}')
                flag = True
        if flag:
            pairs_drone2sate_list_save.append(pairs_drone2sate)

    pairs_iou_match_set = set()
    for sate_img, tile2drone in pairs_iou_sate2drone_dict.items():
        pairs_iou_sate2drone_dict[sate_img] = list(set(tile2drone))
    for drone_img, drone2tile in pairs_iou_drone2sate_dict.items():
        pairs_iou_drone2sate_dict[drone_img] = list(set(drone2tile))
        for sate_img in pairs_iou_drone2sate_dict[drone_img]:
            pairs_iou_match_set.add((drone_img, f'{sate_img}'))
    pairs_semi_iou_match_set = set()
    for sate_img, tile2drone in pairs_semi_iou_sate2drone_dict.items():
        pairs_semi_iou_sate2drone_dict[sate_img] = list(set(tile2drone))
    for drone_img, drone2tile in pairs_semi_iou_drone2sate_dict.items():
        pairs_semi_iou_drone2sate_dict[drone_img] = list(set(drone2tile))
        for sate_img in pairs_semi_iou_drone2sate_dict[drone_img]:
            pairs_semi_iou_match_set.add((drone_img, f'{sate_img}'))

    with open(pkl_save_path, 'wb') as f:
        pickle.dump({
            "pairs_drone2sate_list": pairs_drone2sate_list_save,
            "pairs_iou_sate2drone_dict": pairs_iou_sate2drone_dict,
            "pairs_iou_drone2sate_dict": pairs_iou_drone2sate_dict,
            "pairs_iou_match_set": pairs_iou_match_set,
            "pairs_semi_iou_sate2drone_dict": pairs_semi_iou_sate2drone_dict,
            "pairs_semi_iou_drone2sate_dict": pairs_semi_iou_drone2sate_dict,
            "pairs_semi_iou_match_set": pairs_semi_iou_match_set,
        }, f)


def process_visloc_data(
        root,
        save_root,
        split_type,
        output_prefix=None,
        scene_ids=None,
        train_scene_ids=None,
        test_scene_ids=None,
        same_area_split_mode='legacy',
        train_ratio=0.8,
        split_seed=42,
        num_workers=None,
    ):
    processed_data_train = []
    processed_data_test = []

    os.makedirs(save_root, exist_ok=True)
    output_prefix = output_prefix or split_type
    train_scene_ids, test_scene_ids = configure_scene_lists(
        split_type,
        scene_ids=scene_ids,
        train_scene_ids=train_scene_ids,
        test_scene_ids=test_scene_ids,
    )
    selected_scene_ids = normalize_scene_ids(train_scene_ids + test_scene_ids)
    raw_scene_counts = {format_scene_id(scene_id): 0 for scene_id in selected_scene_ids}
    skipped_scenes = {}

    sate_meta_file_candidates = [
        os.path.join(root, 'satellite_coordinates_range.csv'),
        os.path.join(root, 'satellite_ coordinates_range.csv'),
    ]
    sate_meta_file = next((p for p in sate_meta_file_candidates if os.path.exists(p)), None)
    if sate_meta_file is None:
        raise FileNotFoundError(
            f"Cannot find satellite coordinates CSV under {root}. Tried: {', '.join(os.path.basename(p) for p in sate_meta_file_candidates)}"
        )
    sate_meta_data = {}
    with open(sate_meta_file, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        header = next(csvreader)
        for row in csvreader:
            name_sate = row[0][9: 11]
            sate_meta_data[name_sate] = {
                "LT_lat": float(row[1]),
                "LT_lon": float(row[2]),
                "RB_lat": float(row[3]),
                "RB_lon": float(row[4]),
            }

    train_drone_meta_data_list = []
    test_drone_meta_data_list = []
    drone_meta_data_list = []
    for i in selected_scene_ids:
        str_i = format_scene_id(i)
        file_dir = os.path.join(root, str_i)

        drone_meta_file = os.path.join(file_dir, f'{str_i}.csv')
        sate_tif_path = os.path.join(file_dir, f'satellite{str_i}.tif')
        if not os.path.isfile(drone_meta_file):
            raise FileNotFoundError(f'Missing drone metadata CSV for scene {str_i}: {drone_meta_file}')
        if not os.path.isfile(sate_tif_path):
            raise FileNotFoundError(f'Missing satellite tif for scene {str_i}: {sate_tif_path}')

        sate_img = cv2.imread(sate_tif_path)
        if sate_img is None:
            raise FileNotFoundError(f'Failed to read satellite tif for scene {str_i}: {sate_tif_path}')
        sate_pix_h, sate_pix_w, _ = sate_img.shape

        with open(drone_meta_file, 'r', encoding='utf-8-sig', newline='') as csvfile:
            csvreader = csv.DictReader(csvfile, delimiter=',')
            fieldnames_ci = {str(field_name).lower(): field_name for field_name in (csvreader.fieldnames or [])}
            required_basic_fields = {'filename', 'lat', 'lon', 'height'}
            required_pose_fields = {'omega', 'kappa', 'phi1', 'phi2'}
            missing_basic_fields = sorted(required_basic_fields - set(fieldnames_ci.keys()))
            if missing_basic_fields:
                raise KeyError(f'Scene {str_i} CSV misses required columns: {missing_basic_fields}')
            missing_pose_fields = sorted(required_pose_fields - set(fieldnames_ci.keys()))

            if missing_pose_fields:
                scene_row_count = 0
                for _ in csvreader:
                    scene_row_count += 1
                raw_scene_counts[str_i] = scene_row_count
                skipped_scenes[str_i] = f'missing_pose_columns:{",".join(missing_pose_fields)}'
                print(f'Skip scene {str_i}: missing pose columns {missing_pose_fields}')
                continue

            for row in csvreader:
                tmp_meta_data = (
                    file_dir,
                    str_i,
                    row[fieldnames_ci['filename']],
                    float(row[fieldnames_ci['lat']]),
                    float(row[fieldnames_ci['lon']]),
                    float(row[fieldnames_ci['height']]),
                    float(row[fieldnames_ci['omega']]),
                    float(row[fieldnames_ci['kappa']]),
                    float(row[fieldnames_ci['phi1']]),
                    float(row[fieldnames_ci['phi2']]),
                    sate_meta_data[str_i]["LT_lat"],
                    sate_meta_data[str_i]["LT_lon"],
                    sate_meta_data[str_i]["RB_lat"],
                    sate_meta_data[str_i]["RB_lon"],
                    sate_pix_h,
                    sate_pix_w,
                )
                raw_scene_counts[str_i] += 1

                if split_type == 'cross-area':
                    if i in train_scene_ids:
                        train_drone_meta_data_list.append(tmp_meta_data)
                    else:
                        test_drone_meta_data_list.append(tmp_meta_data)
                else:
                    drone_meta_data_list.append(tmp_meta_data)

    if split_type == 'same-area':
        processed_data_total = process_entries_in_parallel(
            drone_meta_data_list,
            num_workers=num_workers,
            desc=f'{output_prefix} same-area pairs',
        )
        rng = random.Random(int(split_seed))
        if same_area_split_mode == 'legacy':
            processed_data_shuffled = list(processed_data_total)
            rng.shuffle(processed_data_shuffled)
            data_num = len(processed_data_shuffled)
            split_index = data_num // 5 * 4
            processed_data_train = processed_data_shuffled[:split_index]
            processed_data_test = processed_data_shuffled[split_index:]
        elif same_area_split_mode == 'per_scene':
            per_scene_pairs = {}
            for result in processed_data_total:
                per_scene_pairs.setdefault(result['str_i'], []).append(result)
            for scene_id in sorted(per_scene_pairs.keys()):
                scene_pairs = list(per_scene_pairs[scene_id])
                rng.shuffle(scene_pairs)
                split_index = compute_same_area_split_index(len(scene_pairs), train_ratio=train_ratio)
                processed_data_train.extend(scene_pairs[:split_index])
                processed_data_test.extend(scene_pairs[split_index:])
        else:
            raise ValueError(f'Unsupported same_area_split_mode: {same_area_split_mode}')
    
    else:
        processed_data_train = process_entries_in_parallel(
            train_drone_meta_data_list,
            num_workers=num_workers,
            desc=f'{output_prefix} cross-area train pairs',
        )
        processed_data_test = process_entries_in_parallel(
            test_drone_meta_data_list,
            num_workers=num_workers,
            desc=f'{output_prefix} cross-area test pairs',
        )
        processed_data_total = processed_data_train + processed_data_test


    train_pkl_save_path = os.path.join(save_root, 'train_pair_meta.pkl')
    train_data_save_dir = os.path.join(save_root, 'train')
    save_pairs_meta_data(processed_data_train, train_pkl_save_path, train_data_save_dir)

    test_pkl_save_path = os.path.join(save_root, 'test_pair_meta.pkl')
    test_data_save_dir = os.path.join(save_root, 'test')
    save_pairs_meta_data(processed_data_test, test_pkl_save_path, test_data_save_dir)

    write_json(save_root, root, output_prefix)
    split_summary = build_split_summary(
        split_type=split_type,
        output_prefix=output_prefix,
        train_scene_ids=train_scene_ids,
        test_scene_ids=test_scene_ids,
        selected_scene_ids=selected_scene_ids,
        skipped_scenes=skipped_scenes,
        same_area_split_mode=same_area_split_mode,
        train_ratio=train_ratio,
        split_seed=split_seed,
        raw_scene_counts=raw_scene_counts,
        processed_data_total=processed_data_total,
        processed_data_train=processed_data_train,
        processed_data_test=processed_data_test,
        output_root=root,
        save_root=save_root,
    )
    summary_path = os.path.join(root, f'{output_prefix}-split-summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(split_summary, f, indent=4, ensure_ascii=False)
    print(f'Saved split summary to {summary_path}')

def write_json(pickle_root, root, output_prefix):
    for type in ['train', 'test']:
        data_drone2sate_json = []
        with open(os.path.join(pickle_root, f'{type}_pair_meta.pkl'), 'rb') as f:
            data_pickle = pickle.load(f)
        for pair_drone2sate in data_pickle['pairs_drone2sate_list']:
            img_name = pair_drone2sate['drone_img']
            
            data_drone2sate_json.append({
                "drone_img_dir": "drone/images",
                "drone_img_name": pair_drone2sate['drone_img'],
                "drone_loc_lat_lon": (pair_drone2sate['lat'], pair_drone2sate['lon']),
                "sate_img_dir": "satellite",
                "pair_pos_sate_img_list": pair_drone2sate['pair_pos_sate_img_list'],
                "pair_pos_sate_weight_list": pair_drone2sate['pair_pos_sate_weight_list'],
                "pair_pos_sate_loc_lat_lon_list": pair_drone2sate['pair_pos_sate_loc_lat_lon_list'],
                "pair_pos_semipos_sate_img_list": pair_drone2sate['pair_pos_semipos_sate_img_list'],
                "pair_pos_semipos_sate_weight_list": pair_drone2sate['pair_pos_semipos_sate_weight_list'],
                "pair_pos_semipos_sate_loc_lat_lon_list": pair_drone2sate['pair_pos_semipos_sate_loc_lat_lon_list'],
                "drone_metadata": {
                    "height": pair_drone2sate.get('height'),
                    "omega": pair_drone2sate.get('omega'),
                    "kappa": pair_drone2sate.get('kappa'),
                    "phi1": pair_drone2sate.get('phi1'),
                    "phi2": pair_drone2sate.get('phi2'),
                    "drone_roll": None,
                    "drone_pitch": None,
                    "drone_yaw": pair_drone2sate.get('drone_yaw', pair_drone2sate.get('phi1')),
                    "cam_roll": None,
                    "cam_pitch": None,
                    "cam_yaw": pair_drone2sate.get('cam_yaw', pair_drone2sate.get('phi1')),
                }
            })
        save_path = os.path.join(root, f'{output_prefix}-drone2sate-{type}.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data_drone2sate_json, f, indent=4, ensure_ascii=False)

def get_sate_data(root_dir):
    sate_img_dir_list = []
    sate_img_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            sate_img_dir_list.append(root)
            sate_img_list.append(file)
    return sate_img_dir_list, sate_img_list



def parse_args():
    parser = argparse.ArgumentParser(
        description='Pre-process UAV-VisLoc into GTA-UAV style train/test JSON files.'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='/home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset',
        help='Root of UAV-VisLoc dataset',
    )
    parser.add_argument(
        '--save_root',
        type=str,
        default='',
        help='Directory for intermediate pair meta pickle files. Defaults to <data_root>/<output_prefix with hyphen replaced by underscore>.',
    )
    parser.add_argument(
        '--split_type',
        type=str,
        default='same-area',
        choices=('same-area', 'cross-area'),
        help='Which evaluation protocol to build.',
    )
    parser.add_argument(
        '--output_prefix',
        type=str,
        default='',
        help='Prefix for generated JSON files, for example same-area-expanded.',
    )
    parser.add_argument(
        '--scene_ids',
        type=str,
        default='',
        help='Comma-separated scene IDs for same-area, for example 03,04 or all.',
    )
    parser.add_argument(
        '--train_scene_ids',
        type=str,
        default='',
        help='Comma-separated scene IDs for cross-area train split. Default is 03.',
    )
    parser.add_argument(
        '--test_scene_ids',
        type=str,
        default='',
        help='Comma-separated scene IDs for cross-area test split. Default is 04.',
    )
    parser.add_argument(
        '--same_area_split_mode',
        type=str,
        default='legacy',
        choices=('legacy', 'per_scene'),
        help='Split strategy for same-area protocols. legacy keeps the original global shuffle; per_scene uses stratified per-scene split.',
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Train ratio for same-area per-scene split.',
    )
    parser.add_argument(
        '--split_seed',
        type=int,
        default=42,
        help='Random seed for same-area splitting.',
    )
    parser.add_argument(
        '--skip_asset_prep',
        action='store_true',
        help='Skip tile/copy preparation and only build pair metadata files.',
    )
    parser.add_argument(
        '--skip_existing_tiles',
        action='store_true',
        help='When preparing assets, skip scenes that already have a non-empty tile directory.',
    )
    parser.add_argument(
        '--keep_last_n_zooms',
        type=int,
        default=0,
        help='If >0, only generate the last N zoom levels per scene. Use 3 to match the current pair-generation path.',
    )
    parser.add_argument(
        '--max_zoom_stub_only',
        action='store_true',
        help='When limiting zoom generation, keep the highest zoom directory as an empty stub instead of materializing its tiles.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='ProcessPool worker count for pair generation.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    ############################################################
    ## This script is to pre-process UAV-VisLoc into a similar format as GTA-UAV.
    ## Please refer to the original data at https://github.com/IntelliSensing/UAV-VisLoc.
    ############################################################

    args = parse_args()
    root = os.path.abspath(args.data_root)
    output_prefix = args.output_prefix.strip() or args.split_type
    same_area_scene_ids = parse_scene_ids_arg(args.scene_ids)
    cross_area_train_scene_ids = parse_scene_ids_arg(args.train_scene_ids)
    cross_area_test_scene_ids = parse_scene_ids_arg(args.test_scene_ids)
    train_scene_ids, test_scene_ids = configure_scene_lists(
        args.split_type,
        scene_ids=same_area_scene_ids,
        train_scene_ids=cross_area_train_scene_ids,
        test_scene_ids=cross_area_test_scene_ids,
    )
    selected_scene_ids = normalize_scene_ids(train_scene_ids + test_scene_ids)
    save_root = args.save_root.strip()
    if not save_root:
        save_root = os.path.join(root, output_prefix.replace('-', '_'))
    save_root = os.path.abspath(save_root)

    set_scene_globals(train_scene_ids, test_scene_ids)
    ensure_scene_files_exist(root, selected_scene_ids, require_tile=False)

    ################################################################
    ## 1. Split the whole satellite image into tiles and prepare unified image folders
    ################################################################
    if not args.skip_asset_prep:
        tile_satellite(
            root,
            scene_ids=selected_scene_ids,
            skip_existing=args.skip_existing_tiles,
            keep_last_n_zooms=args.keep_last_n_zooms,
            max_zoom_stub_only=args.max_zoom_stub_only,
        )
        copy_satellite(root, scene_ids=selected_scene_ids)
        copy_drone(root, scene_ids=selected_scene_ids)

    ################################################################
    ## 2. Match the drone-view images with satellite tiles and export JSON protocol files
    ################################################################
    ensure_scene_files_exist(root, selected_scene_ids, require_tile=True)
    process_visloc_data(
        root=root,
        save_root=save_root,
        split_type=args.split_type,
        output_prefix=output_prefix,
        scene_ids=same_area_scene_ids,
        train_scene_ids=cross_area_train_scene_ids,
        test_scene_ids=cross_area_test_scene_ids,
        same_area_split_mode=args.same_area_split_mode,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        num_workers=args.num_workers,
    )
