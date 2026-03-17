"""
Urban-GenX | Semantic Interface (Cross-Modal Transformer)
Maps natural language queries → urban scene presets using Sentence-BERT.
Fallback: keyword-based matching if sentence-transformers is not available.

Usage:
    from models.transformer_core import SemanticInterface
    si = SemanticInterface()
    preset = si.query("construction site near highway")
    # Returns: {
    #   'scene_name': 'construction_site',
    #   'dominant_cityscapes_class': 14,  (construction)
    #   'cityscapes_layout': {...},
    #   'acoustic_class': 4,              (drilling)
    #   'acoustic_class_name': 'drilling',
    #   'traffic_multiplier': 1.3,
    #   'noise_level': 0.85,
    #   'description': 'Active construction site...'
    # }
"""

import os
import sys
import numpy as np
import torch

# ── Scene Preset Library ──────────────────────────────────────────────────────
# Each preset defines a "urban scenario" as a distribution over Cityscapes
# label classes and an acoustic class from UrbanSound8K.
#
# Cityscapes label IDs (0-based, gtFine_labelIds):
#   0=unlabeled, 1=ego vehicle, 2=rectification border, 3=out of roi,
#   4=static, 5=dynamic, 6=ground, 7=road, 8=sidewalk, 9=parking,
#   10=rail track, 11=building, 12=wall, 13=fence, 14=guard rail,
#   15=bridge, 16=tunnel, 17=pole, 18=polegroup, 19=traffic light,
#   20=traffic sign, 21=vegetation, 22=terrain, 23=sky, 24=person,
#   25=rider, 26=car, 27=truck, 28=bus, 29=caravan, 30=trailer,
#   31=train, 32=motorcycle, 33=bicycle

# UrbanSound8K class IDs:
#   0=air_conditioner, 1=car_horn, 2=children_playing, 3=dog_bark,
#   4=drilling, 5=engine_idling, 6=gun_shot, 7=jackhammer,
#   8=siren, 9=street_music

SCENE_PRESETS = {
    "construction_site": {
        "description": "Active urban construction site with heavy machinery and roadwork.",
        "anchor_phrases": [
            "construction site", "building site", "roadwork", "excavation",
            "crane", "demolition", "drill", "bulldozer", "scaffolding"
        ],
        "cityscapes_weights": {7: 0.3, 11: 0.25, 12: 0.15, 8: 0.15, 17: 0.1, 23: 0.05},
        "dominant_cityscapes_class": 11,  # building
        "acoustic_class": 4,              # drilling
        "acoustic_class_name": "drilling",
        "traffic_multiplier": 0.6,
        "noise_level": 0.88,
        "green_space": 0.05,
    },
    "busy_intersection": {
        "description": "High-traffic urban intersection with pedestrians and vehicles.",
        "anchor_phrases": [
            "busy intersection", "traffic jam", "crossroads", "junction",
            "congested road", "rush hour", "pedestrian crossing", "highway"
        ],
        "cityscapes_weights": {7: 0.4, 26: 0.25, 8: 0.15, 24: 0.1, 19: 0.05, 23: 0.05},
        "dominant_cityscapes_class": 7,   # road
        "acoustic_class": 1,              # car_horn
        "acoustic_class_name": "car_horn",
        "traffic_multiplier": 1.8,
        "noise_level": 0.75,
        "green_space": 0.08,
    },
    "residential_street": {
        "description": "Quiet residential neighbourhood with low traffic.",
        "anchor_phrases": [
            "residential", "neighbourhood", "suburb", "quiet street",
            "housing", "home", "garden", "apartment", "side street"
        ],
        "cityscapes_weights": {7: 0.2, 11: 0.25, 8: 0.2, 21: 0.15, 22: 0.1, 23: 0.05, 26: 0.05},
        "dominant_cityscapes_class": 8,   # sidewalk
        "acoustic_class": 2,              # children_playing
        "acoustic_class_name": "children_playing",
        "traffic_multiplier": 0.4,
        "noise_level": 0.25,
        "green_space": 0.45,
    },
    "park_green_space": {
        "description": "Urban park with vegetation, open space, and pedestrians.",
        "anchor_phrases": [
            "park", "green space", "garden", "nature", "trees",
            "playground", "recreation", "open space", "greenery"
        ],
        "cityscapes_weights": {21: 0.4, 22: 0.2, 8: 0.15, 23: 0.1, 7: 0.05, 24: 0.1},
        "dominant_cityscapes_class": 21,  # vegetation
        "acoustic_class": 9,              # street_music
        "acoustic_class_name": "street_music",
        "traffic_multiplier": 0.15,
        "noise_level": 0.18,
        "green_space": 0.82,
    },
    "commercial_district": {
        "description": "Commercial area with shops, delivery vehicles, and foot traffic.",
        "anchor_phrases": [
            "commercial", "shopping", "market", "store", "mall",
            "business district", "downtown", "shop", "retail"
        ],
        "cityscapes_weights": {7: 0.3, 11: 0.3, 8: 0.15, 26: 0.1, 20: 0.05, 24: 0.05, 23: 0.05},
        "dominant_cityscapes_class": 11,  # building
        "acoustic_class": 5,              # engine_idling
        "acoustic_class_name": "engine_idling",
        "traffic_multiplier": 1.2,
        "noise_level": 0.62,
        "green_space": 0.12,
    },
    "highway_freeway": {
        "description": "Urban freeway with fast-moving vehicles.",
        "anchor_phrases": [
            "highway", "freeway", "motorway", "expressway", "bypass",
            "ring road", "overpass", "bridge", "fast traffic"
        ],
        "cityscapes_weights": {7: 0.5, 26: 0.2, 27: 0.1, 28: 0.05, 23: 0.1, 14: 0.05},
        "dominant_cityscapes_class": 7,   # road
        "acoustic_class": 5,              # engine_idling
        "acoustic_class_name": "engine_idling",
        "traffic_multiplier": 2.0,
        "noise_level": 0.70,
        "green_space": 0.05,
    },
    "industrial_zone": {
        "description": "Industrial area with factories, warehouses, and heavy machinery.",
        "anchor_phrases": [
            "industrial", "factory", "warehouse", "manufacturing", "plant",
            "depot", "cargo", "freight", "industrial park"
        ],
        "cityscapes_weights": {11: 0.3, 7: 0.25, 12: 0.15, 27: 0.1, 23: 0.1, 22: 0.1},
        "dominant_cityscapes_class": 11,  # building
        "acoustic_class": 7,              # jackhammer
        "acoustic_class_name": "jackhammer",
        "traffic_multiplier": 0.9,
        "noise_level": 0.82,
        "green_space": 0.05,
    },
    "emergency_scene": {
        "description": "Emergency response scene with sirens and blocked roads.",
        "anchor_phrases": [
            "emergency", "accident", "police", "ambulance", "fire truck",
            "siren", "incident", "blocked road", "rescue"
        ],
        "cityscapes_weights": {7: 0.35, 8: 0.2, 26: 0.15, 24: 0.15, 23: 0.1, 19: 0.05},
        "dominant_cityscapes_class": 7,   # road
        "acoustic_class": 8,              # siren
        "acoustic_class_name": "siren",
        "traffic_multiplier": 0.3,
        "noise_level": 0.95,
        "green_space": 0.05,
    },
}

DEFAULT_PRESET = "busy_intersection"


class SemanticInterface:
    """
    Maps natural language queries to urban scene presets.
    Primary: sentence-transformers cosine similarity
    Fallback: keyword matching (works without GPU/SBERT)
    """

    def __init__(self, use_sbert: bool = True, model_name: str = "all-MiniLM-L6-v2"):
        self.use_sbert = False
        self.model = None
        self._preset_names = list(SCENE_PRESETS.keys())
        self._preset_embeddings = None

        if use_sbert:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[SemanticInterface] Loading SBERT model: {model_name}")
                self.model = SentenceTransformer(model_name)
                self._build_preset_embeddings()
                self.use_sbert = True
                print("[SemanticInterface] SBERT loaded ✅")
            except ImportError:
                print("[SemanticInterface] sentence-transformers not found → keyword fallback")
            except Exception as e:
                print(f"[SemanticInterface] SBERT load failed ({e}) → keyword fallback")

    def _build_preset_embeddings(self):
        """Pre-compute embeddings for all anchor phrases of all presets."""
        all_phrases = []
        phrase_to_preset = []
        for name, preset in SCENE_PRESETS.items():
            for phrase in preset["anchor_phrases"]:
                all_phrases.append(phrase)
                phrase_to_preset.append(name)

        embeddings = self.model.encode(all_phrases, convert_to_tensor=True, show_progress_bar=False)
        self._preset_embeddings = embeddings
        self._phrase_to_preset = phrase_to_preset

    def _sbert_match(self, query: str) -> str:
        """Return best-matching preset name using cosine similarity."""
        from sentence_transformers import util as sbert_util
        query_emb = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        scores = sbert_util.cos_sim(query_emb, self._preset_embeddings)[0]
        best_idx = int(scores.argmax())
        return self._phrase_to_preset[best_idx]

    def _keyword_match(self, query: str) -> str:
        """Fallback: simple keyword overlap count."""
        query_lower = query.lower()
        best_name = DEFAULT_PRESET
        best_score = 0
        for name, preset in SCENE_PRESETS.items():
            score = sum(phrase in query_lower for phrase in preset["anchor_phrases"])
            if score > best_score:
                best_score = score
                best_name = name
        return best_name

    def query(self, text: str) -> dict:
        """
        Map a text query to a full urban scene preset.

        Returns dict with:
          scene_name, description, dominant_cityscapes_class,
          cityscapes_weights, acoustic_class, acoustic_class_name,
          traffic_multiplier, noise_level, green_space
        """
        if not text or not text.strip():
            text = "busy city street"

        if self.use_sbert:
            matched_name = self._sbert_match(text)
        else:
            matched_name = self._keyword_match(text)

        preset = dict(SCENE_PRESETS[matched_name])
        preset["scene_name"] = matched_name
        return preset

    def build_condition_tensor(self, preset: dict, img_size: int = 64,
                               num_classes: int = 35, batch_size: int = 1) -> torch.Tensor:
        """
        Convert a scene preset's cityscapes_weights into a one-hot condition tensor.
        Returns: [B, num_classes, img_size, img_size]
        """
        weights = preset.get("cityscapes_weights", {7: 1.0})
        weight_array = np.zeros(num_classes, dtype=np.float32)
        for cls_id, w in weights.items():
            if 0 <= cls_id < num_classes:
                weight_array[cls_id] = w

        # Normalize
        total = weight_array.sum()
        if total > 0:
            weight_array /= total

        # Build spatial layout (use top-weighted classes in spatial regions)
        label_map = np.zeros((img_size, img_size), dtype=np.int64)
        sorted_classes = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        sky_class = weights.get(23, 0)
        road_class = weights.get(7, 0)
        building_class = weights.get(11, 0)

        # Simple layout: sky at top, buildings mid, road at bottom
        sky_row = int(img_size * 0.35)
        building_row = int(img_size * 0.65)

        label_map[:sky_row, :] = 23 if sky_class > 0 else sorted_classes[0][0]
        label_map[sky_row:building_row, :] = 11 if building_class > 0 else (
            sorted_classes[0][0] if len(sorted_classes) > 0 else 0)
        label_map[building_row:, :] = 7 if road_class > 0 else (
            sorted_classes[0][0] if len(sorted_classes) > 0 else 0)

        # Add vehicles if present
        if weights.get(26, 0) > 0.1:
            veh_y = int(img_size * 0.75)
            label_map[veh_y:veh_y+8, 10:30] = 26
            label_map[veh_y:veh_y+8, 35:55] = 26

        # Add sidewalk strip
        if weights.get(8, 0) > 0.1:
            label_map[building_row:building_row+4, :] = 8

        cond = torch.zeros(batch_size, num_classes, img_size, img_size)
        lbl_t = torch.from_numpy(label_map).long()
        for b in range(batch_size):
            cond[b].scatter_(0, lbl_t.unsqueeze(0), 1.0)

        return cond

    @staticmethod
    def list_scenes() -> list:
        return list(SCENE_PRESETS.keys())

    @staticmethod
    def get_preset(scene_name: str) -> dict:
        if scene_name in SCENE_PRESETS:
            p = dict(SCENE_PRESETS[scene_name])
            p["scene_name"] = scene_name
            return p
        return dict(SCENE_PRESETS[DEFAULT_PRESET]) | {"scene_name": DEFAULT_PRESET}
