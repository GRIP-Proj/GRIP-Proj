import os
import sys
sys.path.append(os.getcwd())


from configs.model.model_parameters import BackboneConfig_DiT, BackboneConfig_RDT, HeadConfig



class Val_Config:
    def __init__(self, backbone_type="DiT"):
        self.backbone_type = backbone_type
        if self.backbone_type == 'DiT':
            self.backbone = BackboneConfig_DiT()
        elif self.backbone_type == 'RDT':
            self.backbone = BackboneConfig_RDT()
        else:
            assert False, "Unknown backbone type"
        
        self.head = HeadConfig()
        
        self.train_diffusion_step = 1000
        self.val_diffusion_step = 250



