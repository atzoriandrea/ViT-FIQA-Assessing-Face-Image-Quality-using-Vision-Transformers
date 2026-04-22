from .vit_qs import VisionTransformer

def get_model(name, **kwargs):
    
    mode = kwargs.get("mode", "token")
    num_features = kwargs.get("num_features", 512)
    if name == "vit_t_qs":
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, mode=mode)
    elif name == "vit_s_qs":
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, mode=mode)
    elif name == "vit_b_qs":
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, mode=mode)
    elif name == "vit_l_qs":  # For WebFace42M
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, mode=mode)       
    elif name == "vit_h_qs":  # For WebFace42M
         return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=1024, depth=48,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0, mode=mode)
    else:
        raise ValueError()
