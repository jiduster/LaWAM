"""Verify current pose forwarding while reusing the original action/endpoint response."""
from types import SimpleNamespace
from unittest.mock import patch
import msgpack
import numpy as np
import torch
from torch import nn
from scripts.serve_lawam_pose_action import PoseActionLaWAMPolicy, server


def main():
    for enabled in (False,True):
        policy=PoseActionLaWAMPolicy.__new__(PoseActionLaWAMPolicy)
        head=nn.Linear(8,24)
        backend=SimpleNamespace(flow=SimpleNamespace(pose_encoder=object() if enabled else None),
            fingertip_head=head,model_cfg=SimpleNamespace(fingertip_target_mode='endpoint',
            fingertip_target_scale=.1,fingertip_target_center=(0.,0.,1.3)))
        captured=[]
        def predict(examples,**kwargs):
            captured.extend(examples)
            assert kwargs['return_intermediates']
            return dict(normalized_actions=np.zeros((30,44),np.float32),
                        intermediates={'h_t1_pred':np.zeros((1,4,8),np.float32)})
        policy.model=SimpleNamespace(policy_backend=backend,predict_action=predict)
        with patch.object(server.LaWAMPolicy,'__init__',return_value=None):
            policy.__init__()
        spec=SimpleNamespace(embodiment_id=28,stats_key='dexjoco_bimanual',robot_type='dual_arm')
        policy._resolve_robot_spec=lambda obs:spec
        policy._extract_primary_image=lambda obs:obs['base']
        policy._extract_wrist_images=lambda obs,spec:[obs['wrist_left'],obs['wrist_right']]
        policy._normalize_state=lambda state,spec:state*.5
        policy._maybe_log_action_summary=lambda *args:None
        policy.action_hz=30.;policy.guidance_scale=1.;policy.num_inference_steps=2
        policy.stats_bounds='min_max';policy.norm_stats={'dexjoco_bimanual':{'action':{}}}
        pose=np.zeros(12,np.float32);pose[[3,7,11]]=1
        obs=dict(state=np.ones(46),prompt='take a photograph',object_pose_condition=pose)
        obs.update({key:np.zeros((8,8,3),np.uint8) for key in ('base','wrist_left','wrist_right')})
        # Exercise the actual server's wire codec in both directions.
        obs=msgpack.unpackb(msgpack.packb(obs,default=server._pack_array),object_hook=server._unpack_array)
        with patch.object(server,'_unnormalize_min_max',side_effect=lambda actions,*a,**kw:actions+1):
            response=policy.infer(obs)
        assert policy._current_object_pose.get() is None
        if enabled:
            np.testing.assert_array_equal(captured[0]['object_pose_condition'],pose)
        else:
            assert 'object_pose_condition' not in captured[0]
        np.testing.assert_array_equal(captured[0]['state'],np.ones(46)*.5)
        assert len(captured[0]['wrist_image'])==2
        response=msgpack.unpackb(msgpack.packb(response,default=server._pack_array),object_hook=server._unpack_array)
        assert response['actions'].shape==(30,44) and (response['actions']==1).all()
        assert response['fingertip_positions'].shape==(8,3)
        assert 'object_pose_condition' not in response
        del obs['object_pose_condition']
        if enabled:
            try: policy.infer(obs)
            except KeyError: pass
            else: raise AssertionError('Missing pose must fail')
    print('POSE_SERVER_OK baseline/orange, pose forwarding, robot normalization, wire codec, actions and endpoint response')


if __name__=='__main__':
    main()
