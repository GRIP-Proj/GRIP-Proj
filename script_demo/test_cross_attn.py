from cross_grasp.models.cross_attn import Gripper_AutoFit_Attn
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32


dim = 64
head = 8
model = Gripper_AutoFit_Attn(hidden_size=dim, num_heads=head, in_feat=9, out_feat=12, num_layers=4)

model = model.to(device=device, dtype=dtype)
model.init_weight()

model.train()

B = 2
N_gripper = 128
N_object = 1024

gripper_feat = torch.randn(B, N_gripper, dim, dtype=torch.float32, device='cuda')
object_feat = torch.randn(B, N_object, dim, dtype=torch.float32, device='cuda')
contact_config = torch.randn(B, 9, dtype=torch.float32, device='cuda')

out = model(gripper_feat, object_feat, contact_config)



