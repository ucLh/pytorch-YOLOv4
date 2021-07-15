import torch
import sys
sys.path.append('../')
from tool.darknet2pytorch import Darknet

def load_ckpt(path_to_ckpt):
    ckpt = torch.load(path_to_ckpt)
    return dict(ckpt["state_dict"])


params1 = load_ckpt("../../../Integra/yolov4_inference/models/Yolov4_epoch26.pth")
params2 = load_ckpt("../../../Integra/yolov4_inference/models/Yolov4_epoch15.pth")


for name1 in params1:
    if name1 in params2:
        params2[name1].data.copy_(0.5 * params1[name1].data + 0.5 * params2[name1].data)

model = Darknet('../cfg/yolov4.cfg')

model.eval()
model.load_state_dict(params2)
new_state_dict = {'state_dict': model.state_dict()}
torch.save(new_state_dict, '../checkpoints/trash/Yolov4_15+26.pth')
# torch.save(params2, "../ckpt/effnetb0_final_stage/effnetb0_averaged.pth")
#
# sample = torch.ones([1, 3, 64, 64]).to("cuda:0")
# traced = torch.jit.trace(model, torch.rand((1, 3, 256, 1600)))
# traced.save(f"../ckpt/effnetb0_final_stage/traced_effnetb0_averaged.pth")
print("saved")
