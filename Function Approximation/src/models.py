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
    # TODO (Thành viên 1):
    # Dùng if-elif kiểm tra name.lower() và trả về nn.ReLU(), nn.LeakyReLU(), nn.Tanh(), nn.GELU(), nn.Sigmoid().
    raise NotImplementedError("TODO (Thành viên 1): Hãy hoàn thiện hàm get_activation_function.")


class GaussianMLP(nn.Module):
    """
    Mạng Perceptron Đa Tầng (MLP) cho bài toán hồi quy xấp xỉ hàm Gaussian 2 biến.

    Tham số khởi tạo:
        input_dim: Kích thước đầu vào (mặc định 2 cho x1, x2).
        output_dim: Kích thước đầu ra (mặc định 1 cho y).
        num_hidden_layers: Số tầng ẩn C (khi hidden_units là một số nguyên int).
        hidden_units: Số nơ-ron B trên mỗi tầng (int) HOẶC danh sách nơ-ron từng tầng List[int].
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

        # TODO (Thành viên 1):
        # Bước 1: Lưu các thuộc tính input_dim, output_dim, activation.
        # Bước 2: Chuẩn hóa hidden_units (int -> danh sách).
        # Bước 3: Dùng nn.Sequential để tạo chuỗi các tầng nn.Linear xen kẽ hàm kích hoạt.
        #         Lưu ý: Tầng đầu ra cuối cùng chuyển từ hidden_dim cuối về output_dim và KHÔNG dùng activation.
        # Bước 4: Gọi phương thức _init_weights() để khởi tạo trọng số.
        raise NotImplementedError("TODO (Thành viên 1): Hãy triển khai phương thức __init__ của GaussianMLP.")

    def _init_weights(self):
        """Khởi tạo trọng số cho các tầng Linear theo chuẩn Xavier hoặc Kaiming."""
        # TODO (Thành viên 1):
        # Duyệt qua self.modules(), nếu là nn.Linear:
        # Khởi tạo weight bằng kaiming_normal_ (nếu dùng relu) hoặc xavier_uniform_ (nếu dùng tanh/sigmoid).
        # Khởi tạo bias bằng 0 (nn.init.zeros_).
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền xuôi (Forward pass).
        Tensor x kích thước (batch_size, 2) -> Tensor ŷ kích thước (batch_size, 1).
        """
        # TODO (Thành viên 1):
        # Truyền x qua mạng nơ-ron tuần tự và trả về kết quả dự đoán.
        raise NotImplementedError("TODO (Thành viên 1): Hãy triển khai phương thức forward.")

    def count_parameters(self) -> int:
        """Tính tổng số tham số có thể huấn luyện (trainable parameters) của mô hình."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("--- KIỂM TRA MODULE MODELS (THÀNH VIÊN 1) ---")
