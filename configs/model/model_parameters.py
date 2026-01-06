
class BackboneConfig:
    def __init__(self, backbone_type='DiT'):
        
        if backbone_type == 'DiT':
            self.hidden_size = 64
            self.num_heads = 8
            self.text_size = 512
            self.depth = 28
            self.in_channels = 9
            self.learn_sigma = True
            
            
        elif backbone_type == 'RDT':
            self.hidden_size = 64
            self.num_heads = 8
        
        self.diffuse_step = 1000
        self.val_diffuse_step = 250
            
class HeadConfig:
    def __init__(self):
        self.hidden_size = 64
        self.num_heads = 8
        self.in_feat = 9
        self.out_feat = 12
        self.num_layers = 4


class ModelParameters:
    def __init__(self, backbone_type='DiT'):
        self.backbone_type = backbone_type
        self.backbone = BackboneConfig(backbone_type=self.backbone_type)
        self.head = HeadConfig()
        self.train_diffusion_step = 1000
        self.val_diffusion_step = 250
        
        
        
    
    