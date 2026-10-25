"""
Base framework abstraction providing:
- Pretrained loading (config + normalization stats + weights)
- Action space utilities (dimension, stats, (un)normalization)
- Trainable module discovery helper
Note: No device placement or optimizer concerns handled here (delegated to trainer).
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import PretrainedConfig, PreTrainedModel

from starVLA.model.tools import auto_get_trainable_modules

from starVLA.model.framework.share_tools import dict_to_namespace
from starVLA.model.framework.share_tools import read_mode_config
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


LATENT_WORLD_FRAMEWORK_NAMES = {"lawam"}
LATENT_WORLD_REQUIRED_PREFIXES = (
    "policy_backend.vlm.",
    "policy_backend.lam.",
    "policy_backend.vlm_to_lam.",
    "policy_backend.flow.",
    "policy_backend.act_query",
    "policy_backend.flow_action_query",
)


def validate_full_checkpoint_state_dict(
    *,
    checkpoint_keys: set[str],
    framework_name: str,
    checkpoint_path: str | Path,
    require_vlm_action_queries: bool = True,
) -> dict[str, int]:
    """Validate a latent-world init checkpoint.

    LaWAM training accepts either a complete LaWAM checkpoint or a
    VLM-only initialization checkpoint produced from external VLM weights.
    """
    framework_name = str(framework_name).lower()
    if framework_name not in LATENT_WORLD_FRAMEWORK_NAMES:
        return {}

    prefix_counts = {
        prefix: sum(1 for key in checkpoint_keys if key == prefix or key.startswith(prefix))
        for prefix in LATENT_WORLD_REQUIRED_PREFIXES
    }
    missing_prefixes = [
        prefix for prefix, count in prefix_counts.items()
        if count == 0 and (require_vlm_action_queries or prefix != "policy_backend.act_query")
    ]
    is_vlm_only_checkpoint = (
        prefix_counts["policy_backend.vlm."] > 0
        and all(prefix_counts[prefix] == 0 for prefix in LATENT_WORLD_REQUIRED_PREFIXES if prefix != "policy_backend.vlm.")
    )
    if missing_prefixes and not is_vlm_only_checkpoint:
        raise RuntimeError(
            "[LatentWorldPolicy] invalid init checkpoint: "
            f"missing required key prefixes {missing_prefixes} in `{checkpoint_path}`."
        )
    return prefix_counts


def validate_contact_checkpoint_keys(model_keys: set[str], checkpoint_keys: set[str]) -> None:
    """Inference must never silently discard or randomly initialize contact layers."""
    prefixes = ("policy_backend.flow.contact_encoder.", "policy_backend.flow.DiT.contact_attention.")
    expected = {key for key in model_keys if key.startswith(prefixes)}
    actual = {key for key in checkpoint_keys if key.startswith(prefixes)}
    if expected != actual:
        raise RuntimeError(
            "Contact checkpoint/config mismatch: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}. "
            "Use the config saved with this checkpoint. Baseline-to-contact initialization is training-only."
        )


def validate_pose_checkpoint_keys(model_keys: set[str], checkpoint_keys: set[str]) -> None:
    prefixes = ("policy_backend.interaction_encoder.", "policy_backend.flow.DiT.interaction_attention.",
                "policy_backend.flow.pose_encoder.", "policy_backend.flow.DiT.pose_attention.",
                "policy_backend.plan_pose_encoder.", "policy_backend.vlm_to_lam.pose_adapter.")
    expected = {k for k in model_keys if k.startswith(prefixes)}
    actual = {k for k in checkpoint_keys if k.startswith(prefixes)}
    if expected != actual:
        raise RuntimeError("Pose checkpoint/config mismatch; use the saved config. "
                           "Baseline-to-pose initialization is training-only.")


def log_full_checkpoint_summary(
    *,
    framework_name: str,
    checkpoint_path: str | Path,
    checkpoint_keys: set[str],
    prefix_counts: dict[str, int],
    require_vlm_action_queries: bool = True,
) -> None:
    framework_name = str(framework_name).lower()
    if framework_name not in LATENT_WORLD_FRAMEWORK_NAMES:
        return

    is_full_checkpoint = all(
        prefix_counts[prefix] > 0 for prefix in LATENT_WORLD_REQUIRED_PREFIXES
        if require_vlm_action_queries or prefix != "policy_backend.act_query"
    )
    checkpoint_kind = "Full checkpoint" if is_full_checkpoint else "VLM-only checkpoint"
    logger.info(
        "[LatentWorldPolicy] %s loaded | path=%s | total_keys=%d | %s=%d | %s=%d | %s=%d | %s=%d | %s=%d | %s=%d",
        checkpoint_kind,
        str(checkpoint_path),
        len(checkpoint_keys),
        "policy_backend.vlm.",
        prefix_counts["policy_backend.vlm."],
        "policy_backend.lam.",
        prefix_counts["policy_backend.lam."],
        "policy_backend.vlm_to_lam.",
        prefix_counts["policy_backend.vlm_to_lam."],
        "policy_backend.flow.",
        prefix_counts["policy_backend.flow."],
        "policy_backend.act_query",
        prefix_counts["policy_backend.act_query"],
        "policy_backend.flow_action_query",
        prefix_counts["policy_backend.flow_action_query"],
    )


def _drop_removed_lm_head_unexpected_keys(
    *,
    model_state_dict: dict[str, torch.Tensor],
    checkpoint_state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Drop legacy VLM lm_head keys when the current VLA has removed lm_head."""
    model_keys = set(model_state_dict.keys())
    filtered_state_dict = {}
    dropped_keys = set()
    for key, value in checkpoint_state_dict.items():
        if key not in model_keys and "lm_head" in key.split("."):
            dropped_keys.add(key)
            continue
        filtered_state_dict[key] = value
    return filtered_state_dict, dropped_keys


# PreTrainedModel, AutoModel, PretrainedConfig,  are so good, find sometime to study them
# TODO: merge YAML config with transformer config.

class baseframework(PreTrainedModel):
    """
    Lightweight base class for higher-level VLA model assemblies.
    Subclasses are expected to:
      - Accept a structured config
      - Register components in __init__
      - Use provided helpers for action normalization handling
    """

    def __init__(
        self,
        hf_config = PretrainedConfig()
    ) -> None:
        """
        Initialize base nn.Module. Subclasses add components.
        """
        
        super().__init__(hf_config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        **kwargs,
    ) -> None:
        """
        Restore a model instance from a saved checkpoint.

        Workflow:
            1. Resolve checkpoint path
            2. Load config + dataset normalization statistics
            3. Build model with loaded config
            4. Load state_dict with relaxed missing/unexpected-key handling
            5. Attach normalization stats for later un-normalization

        Args:
            pretrained_checkpoint: Path to .pt file inside run/checkpoints directory.
            **kwargs: Extra constructor overrides passed to subclass.

        Returns:
            baseframework: Instantiated model (left on CPU; caller decides device).

        Raises:
            RuntimeError: If incompatible tensor shapes prevent loading.
            FileNotFoundError: If underlying files are missing (surfaced earlier).
        """
        pretrained_checkpoint = Path(pretrained_checkpoint)
        model_config, norm_stats = read_mode_config(pretrained_checkpoint)  # read config and norm_stats

        config = dict_to_namespace(model_config)
        model_config = config
        model_config.trainer.pretrained_checkpoint = None
        # FrameworkModel = cls(config=model_config, **kwargs) # TODO find cls by config
        from starVLA.model.framework import build_framework  # lazy import to avoid circular dependency
        from starVLA.training.trainer_utils.trainer_tools import apply_training_freeze_policy

        FrameworkModel = build_framework(cfg=model_config)
        FrameworkModel = apply_training_freeze_policy(FrameworkModel, model_config)
        # set for action un-norm
        FrameworkModel.norm_stats = norm_stats
        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")
        current_state_dict = FrameworkModel.state_dict()
        model_keys = set(current_state_dict.keys())
        checkpoint_keys = set(model_state_dict.keys())
        validate_contact_checkpoint_keys(model_keys, checkpoint_keys)
        validate_pose_checkpoint_keys(model_keys, checkpoint_keys)
        framework_name = str(getattr(model_config.framework, "name", "")).lower()
        policy_cfg = getattr(FrameworkModel, "policy_cfg", None)
        require_vlm_action_queries = int(getattr(policy_cfg, "num_vlm_action_queries", 1)) > 0
        prefix_counts = validate_full_checkpoint_state_dict(
            checkpoint_keys=checkpoint_keys,
            framework_name=framework_name,
            checkpoint_path=pretrained_checkpoint,
            require_vlm_action_queries=require_vlm_action_queries,
        )
        model_state_dict, dropped_lm_head_keys = _drop_removed_lm_head_unexpected_keys(
            model_state_dict=current_state_dict,
            checkpoint_state_dict=model_state_dict,
        )
        if dropped_lm_head_keys:
            logger.info(
                "Ignoring removed VLM lm_head keys in checkpoint `%s`: %s",
                pretrained_checkpoint,
                sorted(dropped_lm_head_keys),
            )
            checkpoint_keys = set(model_state_dict.keys())
        try:
            load_result = FrameworkModel.load_state_dict(model_state_dict, strict=False)
        except RuntimeError as e:
            common_keys = model_keys.intersection(checkpoint_keys)
            missing_keys = model_keys - common_keys
            unexpected_keys = checkpoint_keys - common_keys
            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

            raise e
        if load_result.missing_keys:
            logger.warning(f"Missing keys in state_dict: {set(load_result.missing_keys)}")
        if load_result.unexpected_keys:
            logger.warning(f"Unexpected keys in state_dict: {set(load_result.unexpected_keys)}")
        log_full_checkpoint_summary(
            framework_name=framework_name,
            checkpoint_path=pretrained_checkpoint,
            checkpoint_keys=checkpoint_keys,
            prefix_counts=prefix_counts,
            require_vlm_action_queries=require_vlm_action_queries,
        )

        return FrameworkModel

    @property
    def trainable_module_keys(self, max_depth=1) -> List[str]:
        """
        Enumerate trainable submodule names up to a depth.

        Args:
            max_depth: Descent depth when traversing module tree.

        Returns:
            List[str]: Module path names considered trainable.
        """
        keys = auto_get_trainable_modules(self, max_depth=max_depth)  # auto check which modules are trainable
        return keys

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Map normalized actions (≈[-1, 1]) back to original value range.

        Steps:
            - Clamp values to [-1, 1]
            - Threshold channel index 6 to {0,1} (binary semantic)
            - Apply linear scaling for masked dimensions using:
                original = 0.5 * (norm + 1) * (q99 - q01) + q01

        Args:
            normalized_actions: Array shape [T, D] (or chunk length × action_dim).
            action_norm_stats: Dict containing:
                q01 (array-like): Lower percentile (per-dimension).
                q99 (array-like): Upper percentile (per-dimension).
                mask (optional bool array): True => apply de-normalization; False => keep original normalized value.

        Returns:
            np.ndarray: Unnormalized actions (same shape as input).
        """
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    @classmethod
    def get_action_stats(self, unnorm_key=None, norm_stats=None):
        """
        Duplicate stats accessor (retained for backward compatibility).
        # in future, it will own to policy interface and pack as 
        """
        if norm_stats ==None:
            norm_stats = self.norm_stats
        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]
