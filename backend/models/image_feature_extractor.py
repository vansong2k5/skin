"""Image feature extraction utilities for dermatology analysis.

This module glues together a lightweight EfficientNet backbone (via
``torchvision``) and an optional Segment-Anything (SAM) predictor to produce a
concise set of handcrafted features that describe uploaded skin lesion
photographs.  The features are later forwarded to the Gemini API so that the
LLM can take structured visual cues into account when producing its answer.

The implementation is intentionally defensive: if either the EfficientNet or
SAM components are unavailable the extractor gracefully falls back to classical
OpenCV based heuristics while still returning useful metadata about the failure
so callers can debug their setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class SegmentationResult:
    mask: np.ndarray
    method: str
    notes: Optional[str] = None


class ImageFeatureExtractor:
    """Compute high level descriptors for dermatology images."""

    def __init__(
        self,
        device: Optional[str] = None,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = "vit_b",
    ) -> None:
        self.device = device
        self._torch = None
        self._transform = None
        self._efficientnet = None
        self._backbone_warning: Optional[str] = None

        self._sam_predictor = None
        self._sam_status = "not_configured"
        self._sam_checkpoint = sam_checkpoint
        self._sam_model_type = sam_model_type

        self._load_backbone()
        self._load_sam()

    # ------------------------------------------------------------------
    # Initialisation helpers
    def _load_backbone(self) -> None:
        try:
            import torch
            from torchvision import transforms
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        except Exception as exc:  # pragma: no cover - depends on optional deps
            self._backbone_warning = f"torch/torchvision unavailable: {exc}"
            return

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch

        weights = None
        try:  # pragma: no cover - network weights may not be present in tests
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self._transform = weights.transforms()
            model = efficientnet_b0(weights=weights)
        except Exception as exc:
            # Fall back to randomly initialised weights so the pipeline remains
            # usable offline.  We keep a warning to let operators know that
            # embeddings will not be meaningful.
            self._backbone_warning = (
                "EfficientNet pretrained weights unavailable; using random "
                f"initialisation ({exc})."
            )
            self._transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            model = efficientnet_b0(weights=None)

        model.classifier = torch.nn.Identity()
        model.to(self.device)
        model.eval()
        self._efficientnet = model

    def _load_sam(self) -> None:
        checkpoint = self._sam_checkpoint or os.getenv("SAM_CHECKPOINT_PATH")
        model_type = self._sam_model_type or os.getenv("SAM_MODEL_TYPE", "vit_b")
        if not checkpoint:
            self._sam_status = "not_configured"
            return

        try:  # pragma: no cover - optional heavy dependency
            from segment_anything import SamPredictor, sam_model_registry

            sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
            if self._torch is None:
                import torch

                self._torch = torch
                if self.device is None:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
            sam_model.to(self.device)
            self._sam_predictor = SamPredictor(sam_model)
            self._sam_status = "ready"
        except Exception as exc:
            self._sam_status = f"error: {exc}"

    # ------------------------------------------------------------------
    # Segmentation utilities
    def _segment_with_sam(self, image: np.ndarray) -> Optional[SegmentationResult]:
        if self._sam_predictor is None:
            return None

        try:  # pragma: no cover - SAM execution is optional
            self._sam_predictor.set_image(image)
            h, w = image.shape[:2]
            center = np.array([[w / 2.0, h / 2.0]])
            point_labels = np.array([1])
            masks, scores, _ = self._sam_predictor.predict(
                point_coords=center,
                point_labels=point_labels,
                multimask_output=True,
            )
            if masks.size == 0:
                raise RuntimeError("SAM returned no masks")
            idx = int(np.argmax(scores)) if scores is not None else 0
            mask = masks[idx].astype(bool)
            return SegmentationResult(mask=mask, method="sam", notes="high-confidence mask")
        except Exception as exc:
            self._sam_status = f"error: {exc}"
            return None

    def _segment_with_otsu(self, image: np.ndarray) -> SegmentationResult:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        mask = mask.astype(np.uint8)
        if mask.mean() > 127:
            mask = cv2.bitwise_not(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return SegmentationResult(mask=mask.astype(bool), method="otsu")

    def _segment(self, image: np.ndarray) -> SegmentationResult:
        sam_result = self._segment_with_sam(image)
        if sam_result:
            return sam_result
        return self._segment_with_otsu(image)

    # ------------------------------------------------------------------
    # Public API
    def extract_features(self, image: Image.Image) -> Dict[str, Any]:
        image = image.convert("RGB")
        np_image = np.array(image)

        seg = self._segment(np_image)
        mask = seg.mask
        if mask.ndim == 2:
            lesion_mask = mask
        else:
            lesion_mask = mask[..., 0]

        total_pixels = lesion_mask.size
        lesion_pixels = int(np.count_nonzero(lesion_mask)) or 1
        area_ratio = float(lesion_pixels) / float(total_pixels)

        hsv = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)
        lesion_indices = lesion_mask.astype(bool)
        hsv_lesion = hsv[lesion_indices] if np.any(lesion_indices) else hsv.reshape(-1, 3)
        rgb_lesion = np_image[lesion_indices] if np.any(lesion_indices) else np_image.reshape(-1, 3)

        hue_mean = float(np.mean(hsv_lesion[:, 0]) / 180.0)
        hue_std = float(np.std(hsv_lesion[:, 0]) / 180.0)
        saturation_mean = float(np.mean(hsv_lesion[:, 1]) / 255.0)
        value_mean = float(np.mean(hsv_lesion[:, 2]) / 255.0)

        red_mean = float(np.mean(rgb_lesion[:, 0]) / 255.0)
        green_mean = float(np.mean(rgb_lesion[:, 1]) / 255.0)
        blue_mean = float(np.mean(rgb_lesion[:, 2]) / 255.0)
        redness_index = max(0.0, red_mean - (green_mean + blue_mean) / 2.0)

        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_vals = laplacian[lesion_indices] if np.any(lesion_indices) else laplacian.reshape(-1)
        texture_var = float(np.var(laplacian_vals))

        contour_mask = (lesion_mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = 0.0
        area = 0.0
        border_irregularity = 0.0
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(largest))
            perimeter = float(cv2.arcLength(largest, True))
            if area > 1.0:
                border_irregularity = float((perimeter ** 2) / (4.0 * np.pi * area))

        # Simple natural language helpers for the downstream prompt
        color_desc = "màu sắc đồng nhất" if hue_std < 0.05 else "màu sắc không đồng nhất"
        texture_desc = "bề mặt mịn" if texture_var < 150.0 else "bề mặt gồ ghề"
        border_desc = "bờ đều" if border_irregularity < 1.5 else "bờ không đều"

        features: Dict[str, Any] = {
            "color_pattern": {
                "dominant_hue": round(hue_mean, 3),
                "hue_std": round(hue_std, 3),
                "saturation": round(saturation_mean, 3),
                "value": round(value_mean, 3),
                "description": color_desc,
            },
            "redness": {
                "mean_red": round(red_mean, 3),
                "index": round(redness_index, 3),
            },
            "texture": {
                "laplacian_variance": round(texture_var, 3),
                "description": texture_desc,
            },
            "border": {
                "irregularity": round(border_irregularity, 3),
                "perimeter": round(perimeter, 3),
                "area": round(area, 3),
                "description": border_desc,
            },
            "area_ratio": round(area_ratio, 3),
            "segmentation": {
                "method": seg.method,
                "notes": seg.notes,
                "sam_status": self._sam_status,
                "lesion_pixels": lesion_pixels,
                "total_pixels": int(total_pixels),
            },
        }

        if self._efficientnet is not None and self._transform is not None and self._torch is not None:
            try:
                tensor = self._transform(image).unsqueeze(0).to(self.device)
                with self._torch.no_grad():
                    embedding = self._efficientnet(tensor).squeeze().cpu().numpy()
                embedding = embedding.astype(float)
                features["embedding_sample"] = [round(float(x), 4) for x in embedding[:16]]
                features["embedding_norm"] = round(float(np.linalg.norm(embedding)), 4)
            except Exception as exc:  # pragma: no cover - depends on torch env
                self._backbone_warning = (
                    self._backbone_warning or f"EfficientNet inference failed: {exc}"
                )

        features["model"] = {
            "backbone": "torchvision::efficientnet_b0" if self._efficientnet else "none",
            "device": self.device or "cpu",
            "sam_status": self._sam_status,
        }
        if self._backbone_warning:
            features["model"]["warnings"] = [self._backbone_warning]

        return features


@lru_cache(maxsize=1)
def get_feature_extractor() -> ImageFeatureExtractor:
    """Return a cached :class:`ImageFeatureExtractor` instance.

    Using a cached singleton avoids repeated model initialisation on every
    request while still keeping the module importable in environments where the
    heavy ML dependencies are missing.
    """

    sam_checkpoint = os.getenv("SAM_CHECKPOINT_PATH")
    sam_model_type = os.getenv("SAM_MODEL_TYPE", "vit_b")
    return ImageFeatureExtractor(sam_checkpoint=sam_checkpoint, sam_model_type=sam_model_type)
