"""Filtro Ignacio v1: aproximación de muñeca al bolsillo lateral (esqueleto 8 KP)."""

from pocket_filter.evaluate import PocketConfig, evaluate_pocket, pocket_result_to_dict
from pocket_filter.io import load_poses_array

__all__ = ["PocketConfig", "evaluate_pocket", "pocket_result_to_dict", "load_poses_array"]