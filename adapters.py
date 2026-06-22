import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class BaseGridAdapter(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.apply_blur = kwargs.get("apply_blur", False)
        self.blur_prob = kwargs.get("blur_prob", 0.5)
        self.blur_kernel_size = kwargs.get("blur_kernel_size", 5)
        self.blur_sigma = kwargs.get("blur_sigma", [0.1, 2.0])

        if self.blur_kernel_size % 2 == 0:
            self.blur_kernel_size += 1

        if isinstance(self.blur_sigma, (int, float)):
            self.blur_sigma = [float(self.blur_sigma), float(self.blur_sigma)]

    def reconstruct(self):
        return self.get_delta_tensor().detach().cpu().numpy()

    def forward(self, x):
        delta = self.get_delta_tensor()
        self._penalty_l1 = torch.sum(torch.abs(delta))
        self._penalty_l2 = torch.sum(delta ** 2)

        if self.apply_blur and self.training:
            if torch.rand(1).item() < self.blur_prob:
                sigma = torch.empty(1).uniform_(self.blur_sigma[0], self.blur_sigma[1]).item()
                delta = TF.gaussian_blur(
                    delta.unsqueeze(0).unsqueeze(0),
                    kernel_size=[self.blur_kernel_size, self.blur_kernel_size],
                    sigma=[sigma, sigma],
                ).squeeze(0).squeeze(0)

        return F.linear(x, delta)

    def get_delta_tensor(self):
        with torch.no_grad():
            return self._generate_weights().detach()


class LoraAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, **kwargs):
        super().__init__()
        self.down_project = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up_project = nn.Linear(hidden_dim, out_dim, bias=False)
        nn.init.zeros_(self.up_project.weight)

        self.apply_blur = kwargs.get("apply_blur", False)
        self.blur_prob = kwargs.get("blur_prob", 0.5)
        self.blur_kernel_size = kwargs.get("blur_kernel_size", 5)
        self.blur_sigma = kwargs.get("blur_sigma", [0.1, 2.0])
        if self.blur_kernel_size % 2 == 0:
            self.blur_kernel_size += 1
        if isinstance(self.blur_sigma, (int, float)):
            self.blur_sigma = [float(self.blur_sigma), float(self.blur_sigma)]

        self.reset_parameters()

    def reconstruct(self):
        return self.get_delta_tensor().detach().cpu().numpy()

    def forward(self, x):
        delta = self.get_delta_tensor()
        self._penalty_l1 = torch.sum(torch.abs(delta))
        self._penalty_l2 = torch.sum(delta ** 2)

        if self.apply_blur and self.training:
            if torch.rand(1).item() < self.blur_prob:
                sigma = torch.empty(1).uniform_(self.blur_sigma[0], self.blur_sigma[1]).item()
                delta = TF.gaussian_blur(
                    delta.unsqueeze(0).unsqueeze(0),
                    [self.blur_kernel_size, self.blur_kernel_size],
                    [sigma, sigma],
                ).squeeze(0).squeeze(0)

        return F.linear(x, delta)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down_project.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_project.weight)

    def get_delta_tensor(self):
        return self.up_project.weight @ self.down_project.weight


class FastGaussianSplattingAdapter(BaseGridAdapter):
    """
    Separable 2D Gaussian splatting adapter.

    DeltaW[y, x] = sum_k amp_k * gy[y, k] * gx[x, k]
    """

    def __init__(self, input_dim, hidden_dim, out_dim, num_gaussians=1228, **kwargs):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_gaussians = kwargs.get("num_gaussians", hidden_dim if hidden_dim > 0 else num_gaussians)

        self.means = nn.Parameter(torch.empty(self.num_gaussians, 2))
        self.log_scales = nn.Parameter(torch.empty(self.num_gaussians, 2))
        self.amplitudes = nn.Parameter(torch.empty(self.num_gaussians))

        self.reset_parameters()

    def _generate_weights(self):
        device = self.means.device
        dtype = self.means.dtype

        x = torch.linspace(-1.0, 1.0, steps=self.input_dim, device=device, dtype=dtype).unsqueeze(1)
        y = torch.linspace(-1.0, 1.0, steps=self.out_dim, device=device, dtype=dtype).unsqueeze(1)

        mx = self.means[:, 0].unsqueeze(0)
        my = self.means[:, 1].unsqueeze(0)

        log_sx = torch.clamp(self.log_scales[:, 0], min=-8.0, max=2.0).unsqueeze(0)
        log_sy = torch.clamp(self.log_scales[:, 1], min=-8.0, max=2.0).unsqueeze(0)

        inv_var_x = torch.exp(-2.0 * log_sx)
        inv_var_y = torch.exp(-2.0 * log_sy)

        gx = torch.exp(-0.5 * (x - mx).pow(2) * inv_var_x)
        gy = torch.exp(-0.5 * (y - my).pow(2) * inv_var_y)

        return (gy * self.amplitudes.unsqueeze(0)) @ gx.T

    def reset_parameters(self):
        nn.init.uniform_(self.means, -1.0, 1.0)
        nn.init.constant_(self.log_scales, -4)
        nn.init.zeros_(self.amplitudes)


class LayerWithAdapter(nn.Module):
    def __init__(self, original_layer, adapter_classes, adapter_dim=32, **kwargs):
        super().__init__()
        self.original_layer = original_layer
        self.register_buffer("initial_base_weight", original_layer.weight.detach().clone())
        self.original_layer.requires_grad_(False)
        self.adapters = nn.ModuleList()

        self.l1_lambda = kwargs.get("l1_lambda", 0.0)
        self.l2_lambda = kwargs.get("l2_lambda", 0.0)

        adapter_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"l1_lambda", "l2_lambda", "trainable_base", "target_strategy", "name", "lr", "muon_learning_rate"}
        }

        for adapter_cls in adapter_classes:
            self.adapters.append(
                adapter_cls(
                    original_layer.in_features,
                    adapter_dim,
                    original_layer.out_features,
                    **adapter_kwargs,
                )
            )

    def get_regularization_loss(self):
        if self.l1_lambda == 0.0 and self.l2_lambda == 0.0:
            return 0.0

        reg_loss = 0.0
        for adapter in self.adapters:
            if hasattr(adapter, "_penalty_l1") and self.l1_lambda > 0.0:
                reg_loss += self.l1_lambda * adapter._penalty_l1
            if hasattr(adapter, "_penalty_l2") and self.l2_lambda > 0.0:
                reg_loss += self.l2_lambda * adapter._penalty_l2
        return reg_loss

    def get_adapter_weight(self):
        weights = {f"{adapter.__class__.__name__}_{i}": adapter.reconstruct() for i, adapter in enumerate(self.adapters)}

        if self.original_layer.weight.requires_grad:
            with torch.no_grad():
                weights["Base_Finetuning_Delta"] = (
                    self.original_layer.weight - self.initial_base_weight
                ).cpu().numpy()

        return weights

    def forward(self, x):
        output = self.original_layer(x)
        for adapter in self.adapters:
            output = output + adapter(x)
        return output