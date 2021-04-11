import torch

dummy_input = torch.randn(1, 3, 224, 224)
state_dict = torch.load(r'./mobilenetv2_jester_online.pth.tar')

from mobilenet_v2_tsm import MobileNetV2
torch_module = MobileNetV2(n_class=27)
torch_module.load_state_dict(torch.load("mobilenetv2_jester_online.pth.tar"))
torch_module.eval()
torch_module.load_state_dict(state_dict)
shift_buffer = [torch.zeros([1, 3, 56, 56]),
                torch.zeros([1, 4, 28, 28]),
                torch.zeros([1, 4, 28, 28]),
                torch.zeros([1, 8, 14, 14]),
                torch.zeros([1, 8, 14, 14]),
                torch.zeros([1, 8, 14, 14]),
                torch.zeros([1, 12, 14, 14]),
                torch.zeros([1, 12, 14, 14]),
                torch.zeros([1, 20, 7, 7]),
                torch.zeros([1, 20, 7, 7])]
# 因为需要偏移缓存区，所以有dummy_input, *shift_buffer两个输入，需要把它们组合成元组变成一个输入，
# 另外，还需加上opset_version=10，该方案来自：https://github.com/pytorch/fairseq/issues/1669#issuecomment-798972533
torch.onnx.export(torch_module, (dummy_input, *shift_buffer), "mobilenet_v2_tsm.onnx", opset_version=10, verbose=True)