"""Preserve the official DexJoCo serving path and pass live Oracle interaction data."""
from contextvars import ContextVar

import numpy as np
import tyro
from dexjoco_openpi_client import lawam_policy_server as server
from starVLA.interaction_geometry import SCHEMA, validate_geometry


class InteractionLaWAMPolicy(server.LaWAMPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._interaction_payload=ContextVar('lawam_interaction_payload',default=None)
        original=self.model.predict_action
        oracle=self.model.policy_cfg.interaction_mode=='oracle'
        def predict(examples,**kwargs):
            if oracle:
                payload=self._interaction_payload.get()
                if payload is None: raise KeyError('Oracle requires synchronized interaction geometry')
                value,valid=payload
                examples=[dict(ex,interaction_geometry=value.copy(),interaction_valid=valid.copy()) for ex in examples]
            return original(examples,**kwargs)
        self.model.predict_action=predict

    def infer(self,obs):
        cfg=self.model.policy_cfg
        payload=None
        if cfg.interaction_mode=='oracle':
            if obs.get('interaction_schema')!=SCHEMA or obs.get('interaction_task')!=cfg.interaction_task:
                raise ValueError('Oracle interaction schema/task mismatch; use scripts/eval_lawam_interaction.py')
            value=np.asarray(obs['interaction_geometry'],dtype=np.float32)
            valid=np.asarray(obs['interaction_valid'])
            validate_geometry(value,valid,cfg.interaction_task)
            if value.ndim!=2: raise ValueError('Expected one current interaction frame [E,F]')
            payload=(value.copy(),valid.copy())
        token=self._interaction_payload.set(payload)
        try: return super().infer(obs)
        finally: self._interaction_payload.reset(token)


if __name__=='__main__':
    server.LaWAMPolicy=InteractionLaWAMPolicy
    tyro.cli(server.serve_lawam)
