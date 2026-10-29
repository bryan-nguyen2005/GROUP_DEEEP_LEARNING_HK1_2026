"""
================================================================================
KHUNG SƯỜN MÃ NGUỒN (SKELETON TEMPLATE)
MODULE: models.py
PHỤ TRÁCH: THÀNH VIÊN 1 (Data & Architecture Lead)
MÔ TẢ NHIỆM VỤ:
    1. Cài đặt hàm ánh xạ tên hàm kích hoạt sang torch.nn (ReLU làm baseline, LeakyReLU, Tanh, Sigmoid).
    2. Cài đặt lớp mạng MLP tổng quát kế thừa torch.nn.Module:
        - Tùy chỉnh số tầng ẩn C (num_hidden_layers).
        - Tùy chỉnh số nơ-ron B (hidden_units) trên mỗi tầng.
        - Tùy chỉnh hàm kích hoạt (activation), mặc định 'relu'.
        - Có khởi tạo trọng số chuẩn (Xavier hoặc Kaiming).
    3. Cài đặt hàm forward pass nhận đầu vào (batch_size, 2) và trả về (batch_size, 1).
================================================================================
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from typing import Union, List


def get_activation_function(name: str) -> nn.Module:
    """
    Trả về module hàm kích hoạt của PyTorch tương ứng với tên gọi.
    Hỗ trợ: 'relu' (baseline), 'leaky_relu', 'tanh', 'gelu', 'sigmoid'.
    """
    activations = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
    }
    if not isinstance(name, str) or name.lower() not in activations:
        raise ValueError(f"Hàm kích hoạt không được hỗ trợ: {name!r}.")
    return activations[name.lower()]()


class GaussianMLP(nn.Module):
    """
    Mạng Perceptron Đa Tầng (MLP) cho bài toán hồi quy xấp xỉ hàm Gaussian 2 biến.

    Tham số khởi tạo:
        input_dim: Kích thước đầu vào (mặc định 2 cho x1, x2).
        output_dim: Kích thước đầu ra (mặc định 1 cho y).
        num_hidden_layers: Số tầng ẩn C (khi hidden_units là một số nguyên int).
        hidden_units: Số nơ-ron B trên mỗi tầng (int) HOẶC danh sách nơ-ron từng tầng List[int].
            Nếu là danh sách, độ dài danh sách quyết định số tầng ẩn.
        activation: Tên hàm kích hoạt (mặc định 'relu').
        use_bias: Có sử dụng vector bias hay không (mặc định True).
    """
    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        num_hidden_layers: int = 1,
        hidden_units: Union[int, List[int]] = 64,
        activation: str = "relu",
        use_bias: bool = True
    ):
        super(GaussianMLP, self).__init__()

        def positive_int(value):
            return isinstance(value, int) and not isinstance(value, bool) and value > 0

        if not positive_int(input_dim) or not positive_int(output_dim):
            raise ValueError("input_dim và output_dim phải là số nguyên dương.")
        if not positive_int(num_hidden_layers):
            raise ValueError("num_hidden_layers phải là số nguyên dương.")
        if isinstance(hidden_units, int) and not isinstance(hidden_units, bool):
            widths = [hidden_units] * num_hidden_layers
        elif isinstance(hidden_units, list):
            widths = hidden_units.copy()
        else:
            raise ValueError("hidden_units phải là số nguyên hoặc danh sách số nguyên.")
        if not widths or not all(positive_int(width) for width in widths):
            raise ValueError("Mỗi tầng ẩn phải có số nơ-ron nguyên dương.")
        get_activation_function(activation)  # Kiểm tra tên trước khi tạo mạng.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation.lower()
        self.hidden_units = widths
        self.num_hidden_layers = len(widths)

        layers = []
        previous = input_dim
        for width in widths:
            layers.append(nn.Linear(previous, width, bias=use_bias))
            layers.append(get_activation_function(self.activation))
            previous = width
        layers.append(nn.Linear(previous, output_dim, bias=use_bias))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Khởi tạo trọng số cho các tầng Linear theo chuẩn Xavier hoặc Kaiming."""
        linear_layers = [layer for layer in self.network if isinstance(layer, nn.Linear)]
        for layer in linear_layers[:-1]:
            if self.activation in ("relu", "leaky_relu"):
                nn.init.kaiming_normal_(layer.weight, a=0.01 if self.activation == "leaky_relu" else 0,
                                        nonlinearity="leaky_relu" if self.activation == "leaky_relu" else "relu")
            else:
                nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        # Tầng đầu ra tuyến tính dùng Xavier, không phụ thuộc activation của tầng ẩn.
        nn.init.xavier_uniform_(linear_layers[-1].weight)
        if linear_layers[-1].bias is not None:
            nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền xuôi (Forward pass).
        Tensor x kích thước (batch_size, 2) -> Tensor ŷ kích thước (batch_size, 1).
        """
        return self.network(x)

    def count_parameters(self) -> int:
        """Tính tổng số tham số có thể huấn luyện (trainable parameters) của mô hình."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("--- KIỂM TRA MODULE MODELS (THÀNH VIÊN 1) ---")
