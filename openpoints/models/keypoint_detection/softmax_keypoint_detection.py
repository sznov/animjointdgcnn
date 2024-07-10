import torch
import torch.nn as nn
from ..build import MODELS, build_model_from_cfg

@MODELS.register_module()
class SoftmaxKeypointDetection(nn.Module):
    def __init__(self,
                 num_keypoints=None,
                 encoder_args=None,
                 decoder_args=None,
                 cls_args=None,
                 longrange_args=None,
                 **kwargs):
        super().__init__()

        self.encoder = build_model_from_cfg(encoder_args)
        self.decoder = None
        self.head = None
        
        if encoder_args.NAME in ['DGCNN']:
            self.features_to_point_keypoint_pair_scalars = nn.Linear(encoder_args.embed_dim, num_keypoints)
        else:
            raise NotImplementedError(f'Encoder {encoder_args.NAME} not supported')

    def forward(self, data):
        p = data['pos']
        f = self.encoder.forward(data)
        point_keypoint_pair_scalars = self.features_to_point_keypoint_pair_scalars(f.transpose(-1, -2))
        
        # TODO Experimental
        # Similar to the above, but with a different way of filtering top K points
        # K will now depend on the keypoint type
        # num_keypoints = point_keypoint_pair_scalars.shape[-1]
        # batch_size = point_keypoint_pair_scalars.shape[0]
        # device = point_keypoint_pair_scalars.device
        # keypoints = torch.zeros(batch_size, num_keypoints, 3).to(device)
        # for i in range(num_keypoints):
        #     k = p.shape[1] // 4
        #     if i in [0, 5]: # pelvis, spine
        #         k = p.shape[1] // 4
        #     if i in [6, 31]: # clavicle
        #         k = p.shape[1] // 4
        #     if i in [7, 32]: # upper arm
        #         k = p.shape[1] // 4
        #     if i in [8, 33]: # lower arm
        #         k = p.shape[1] // 4
        #     if i in range(9, 11 + 20) or i in range(34, 36 + 20): # hand, fingers
        #         k = p.shape[1] // 32
        #     if i in [56, 58]: # neck, head 
        #         k = p.shape[1] // 4
        #     if i in [61, 62, 63, 66, 67, 68]: # foot, ball
        #         k = p.shape[1] // 16

        #     topk, topk_indices = torch.topk(point_keypoint_pair_scalars[:, :, i], k, dim=1)
        #     topk_indices_exp = topk_indices.unsqueeze(2).expand(-1,-1,3)
        #     q=torch.gather(p, dim=1, index=topk_indices_exp)
        #     topk = torch.softmax(topk, dim=1)
        #     topk_exp = topk.unsqueeze(2).expand(-1, -1, 3)
        #     q_w = q * topk_exp
        #     keypoints[:, i] = torch.sum(q_w, dim=1)
        # return keypoints

        # Derive keypoint locations as convex combinations of the points
        attention_weights = torch.softmax(point_keypoint_pair_scalars, dim=1)
        keypoints = attention_weights.transpose(-1, -2) @ p

        # DEBUG Visualization of attention weights
        # import numpy as np
        # aw = attention_weights[0].detach().cpu().numpy()
        # np.save('attention_weights.npy', aw)

        return keypoints