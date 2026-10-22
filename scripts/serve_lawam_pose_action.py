"""Preserve the DexJoCo server's action/endpoint response while passing current pose.

CLI is identical to dexjoco_openpi_client.lawam_policy_server. Supports both
matched baseline and orange checkpoints; the baseline ignores the extra field.
"""
from contextvars import ContextVar

import tyro
from dexjoco_openpi_client import lawam_policy_server as server
from starVLA.object_pose import validate_object_pose


class PoseActionLaWAMPolicy(server.LaWAMPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_object_pose = ContextVar('lawam_object_pose', default=None)
        original_predict = self.model.predict_action
        enabled = self.model.policy_backend.flow.pose_encoder is not None

        def predict_with_pose(examples, **predict_kwargs):
            if enabled:
                pose = self._current_object_pose.get()
                if pose is None:
                    raise KeyError('Pose policy requires current object_pose_condition')
                examples = [dict(example, object_pose_condition=pose.copy()) for example in examples]
            return original_predict(examples, **predict_kwargs)

        # The wrapper calls the original server infer(), including endpoint decoding.
        self.model.predict_action = predict_with_pose

    def infer(self, obs):
        enabled = self.model.policy_backend.flow.pose_encoder is not None
        pose = None
        if enabled:
            if 'object_pose_condition' not in obs:
                raise KeyError('Pose policy requires live object_pose_condition')
            pose = validate_object_pose(obs['object_pose_condition']).copy()
            if pose.shape != (12,):
                raise ValueError('One observation requires object_pose_condition[12]')
        token = self._current_object_pose.set(pose)
        try:
            return super().infer(obs)
        finally:
            self._current_object_pose.reset(token)


if __name__ == '__main__':
    server.LaWAMPolicy = PoseActionLaWAMPolicy
    tyro.cli(server.serve_lawam)
