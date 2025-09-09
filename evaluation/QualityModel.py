import os
import sys


sys.path.append('../')
print(os.getcwd())
from FaceModel import FaceModel

from backbones.iresnet_qs import iresnet100, iresnet50
import torch
from backbones.vit_qs import VisionTransformer

import numpy as np


class QualityModel(FaceModel):
    def __init__(self, model_prefix, model_epoch, gpu_id,backbone):
        super(QualityModel, self).__init__(model_prefix, model_epoch, gpu_id,backbone)

    def _get_model(self, ctx, image_size, prefix, epoch, layer, backbone):
        print(backbone)
        if (backbone=="iresnet50" or backbone=="iresnet50_FC"):
            backbones = iresnet50(num_features=512, qs=1, use_se=False).to(f"cuda:{ctx}")
        elif (backbone=="iresnet100"):
            backbones = iresnet100(num_features=512, qs=1, use_se=False).to(f"cuda:{ctx}")
        elif (backbone=="vit_FC"):
            #backbones =VisionTransformer(
            #img_size=112, patch_size=8, num_classes=384, embed_dim=384, depth=12,
            #num_heads=6, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)
            backbones = VisionTransformer(
                img_size=112, patch_size=9, num_classes=512, embed_dim=512, depth=12,
                num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, mode="token")
    
        else:
            backbones = VisionTransformer(img_size=112, patch_size=8, num_classes=512, embed_dim=512, depth=12,
                                    mlp_ratio=5, num_heads=8, drop_path_rate=0.1, norm_layer="ln",
                                    mask_ratio=0.0).to(f"cuda:{ctx}")
        if (backbone=="vit_FC" or backbone=="iresnet50_FC"):
            dict_checkpoint = torch.load(os.path.join(prefix,"model.pt"))
            print(dict_checkpoint.keys)
            for key, value in dict_checkpoint.items() :
                print (key)
            backbones.load_state_dict(dict_checkpoint)
        else:
            weight = torch.load(os.path.join(prefix,epoch+"backbone.pth"))
            backbones.load_state_dict(weight)
        model = torch.nn.DataParallel(backbones, device_ids=[ctx])
        model.eval()
        return model

    @torch.no_grad()
    def _getFeatureBlob(self,input_blob):
        imgs = torch.Tensor(input_blob).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat, qs = self.model(imgs)
        return feat.cpu().numpy(), qs.cpu().numpy() #* np.linalg.norm(feat.cpu().numpy())
