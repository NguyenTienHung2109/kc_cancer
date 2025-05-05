from typing import Tuple
import torch
import onnxruntime
import torch.nn as nn
import torch.nn.functional as F

class InferSliceModule(nn.Module):
    def __init__(self, 
                input_shape: Tuple[int, int, int],
                seg_net: nn.Module, 
                use_lung_pos: bool = False,
                cls_lung_pos_net: nn.Module | None = None,
                use_lung_loc: bool = False,
                get_fm: bool = False):
        
        super().__init__()
        self.input_shape = input_shape
        self.seg_net = seg_net
        self.use_lung_pos = use_lung_pos
        self.use_lung_loc = use_lung_loc
        self.get_fm = get_fm

        if use_lung_pos:
            self.cls_lung_pos_net = cls_lung_pos_net

    def seg_postprocess(self, logits: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            logits (torch.Tensor): (1, c, h, w)
        Returns:
            torch.Tensor: (1, c, h, w) - torch.int
        """
        if logits.shape[1] == 1: # binary segmentation
            probs = torch.sigmoid(logits)
            pred_masks = (probs > 0.5).to(torch.uint8)
        else:
            probs = F.softmax(logits, dim=1)
            pred_masks = torch.argmax(probs, dim=1)
            pred_masks = F.one_hot(pred_masks, num_classes=probs.shape[1]).permute(0, 3, 1, 2)
        return pred_masks.detach().cpu()

    def forward(self, images):
        out = self.seg_net(images, get_fm=True)
        
        res = (self.seg_postprocess(out["seg_nodule"][0]),)
        
        if self.use_lung_loc:
            res = res + (self.seg_postprocess(out["seg_lung_loc"][0]),)
        
        if self.use_lung_pos:
            cls_lung_pos = self.cls_lung_pos_net(out["fm"].mean(dim=[2, 3]))
            res = res + (cls_lung_pos[0], cls_lung_pos[1],)
        
        if self.get_fm:
            scale = images.shape[2] // out["fm"].shape[2]
            scale = scale.item() if isinstance(scale, torch.Tensor) else scale
            fm = F.interpolate(out["fm"], scale_factor=scale, mode='bilinear')
            res = res + (fm,)
        
        return res

    def export_onnx(self, seg_onnx_path: str):
        dummy_input = torch.rand((1, *self.input_shape), device=next(self.parameters()).device)
        output_names = ["seg_nodule"]
        dynamic_axes = {"norm_ct_image": {0: "batch_size"}, 
                        "seg_nodule": {0: "batch_size"}}
        if self.use_lung_loc:
            output_names.append("seg_lung_loc")
            dynamic_axes["seg_lung_loc"] = {0: "batch_size"}
        if self.use_lung_pos:
            output_names.append("right_lung")
            output_names.append("left_lung")
            dynamic_axes["right_lung"] = {0: "batch_size"}
            dynamic_axes["left_lung"] = {0: "batch_size"}
        if self.get_fm:
            output_names.append("fm")
            dynamic_axes["fm"] = {0: "batch_size"}

        torch.onnx.export(self,
                        dummy_input,
                        seg_onnx_path, 
                        export_params=True,
                        input_names=["norm_ct_image"],
                        output_names=output_names,
                        dynamic_axes=dynamic_axes)

    def test_onnx(self, onnx_path: str, device="gpu"):
        dummy_input = torch.rand((1, *self.input_shape))
        providers = ["CUDAExecutionProvider"] if device == "gpu" else ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        
        torch_outs = self.forward(dummy_input.to(next(self.parameters()).device))
        for i, (ort_out, torch_out) in enumerate(zip(ort_outs, torch_outs)):
            print(ort_out.sum(), torch_out.sum(), ort_out.shape, torch_out.shape)
            assert torch.allclose(torch.tensor(ort_out), torch_out.cpu(), rtol=1e-3, atol=1e-3), f"Output {i} mismatch"


class InferNoduleModule(nn.Module):
    def __init__(self, 
                input_size: int, 
                cls_nodule_net):
        super().__init__()
        self.input_size = input_size
        self.cls_nodule_net = cls_nodule_net

    def forward(self, x):
        out = self.cls_nodule_net(x)
        return out
    
    def export_onnx(self, seg_onnx_path: str):
        dummy_input = torch.rand((1, self.input_size), device=next(self.parameters()).device)
        torch.onnx.export(self, 
                        dummy_input, 
                        seg_onnx_path, 
                        export_params=True,
                        input_names=["nodule"], 
                        output_names=["dong_dac", "kinh_mo", "phe_quan_do", "nu_tren_canh", \
                                    "dam_do", "voi_hoa", "chua_mo", "duong_vien", "tao_hang", "di_can"],
                        dynamic_axes={"nodule": {0: "batch_size"},
                                    "dong_dac": {0: "batch_size"},
                                    "kinh_mo": {0: "batch_size"},
                                    "phe_quan_do": {0: "batch_size"},
                                    "nu_tren_canh": {0: "batch_size"},
                                    "dam_do": {0: "batch_size"},
                                    "voi_hoa": {0: "batch_size"},
                                    "chua_mo": {0: "batch_size"},
                                    "duong_vien": {0: "batch_size"},
                                    "tao_hang": {0: "batch_size"},
                                    "di_can": {0: "batch_size"}})

    def test_onnx(self, onnx_path: str, device="gpu"):
        dummy_input = torch.rand((1, self.input_size))
        providers = ["CUDAExecutionProvider"] if device == "gpu" else ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        
        torch_outs = self.forward(dummy_input.to(next(self.parameters()).device))
        for i, (ort_out, torch_out) in enumerate(zip(ort_outs, torch_outs)):
            print(ort_out.sum(), torch_out.sum(), ort_out.shape, torch_out.shape)
            assert torch.allclose(torch.tensor(ort_out), torch_out.cpu(), rtol=1e-3, atol=1e-3), f"Output {i} mismatch"

