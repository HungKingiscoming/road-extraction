"""Smoke test DCNv4 -- chạy TRƯỚC khi đụng vào decoder.py/model.py.

Chạy: python3 smoke_test_dcnv4.py
Nếu bất kỳ bước nào lỗi, dừng lại ở đây -- đừng tích hợp vào training pipeline
cho tới khi cả forward VÀ backward đều chạy sạch.
"""

import subprocess
import sys

print("=" * 70)
print("BƯỚC 1: pip install DCNv4 (build CUDA extension từ source)")
print("=" * 70)
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "DCNv4", "--no-cache-dir"],
    capture_output=True,
    text=True,
)
print(result.stdout[-3000:])
if result.returncode != 0:
    print(result.stderr[-3000:])
    print("\n>>> CÀI ĐẶT THẤT BẠI. Dừng ở đây, đừng tích hợp vào code chính.")
    sys.exit(1)
print(">>> Cài đặt OK.\n")

print("=" * 70)
print("BƯỚC 2: import + forward + backward tối thiểu")
print("=" * 70)
try:
    import torch
    from DCNv4.modules.dcnv4 import DCNv4

    assert torch.cuda.is_available(), "Cần GPU để test DCNv4"
    device = torch.device("cuda")

    # Mô phỏng đúng calling convention thật: input phẳng (N, L, C), NHWC,
    # kèm shape=(N, H, W, C) để module tự suy ra layout không gian.
    N, H, W, C = 2, 32, 32, 32
    # Bằng chứng từ paper DCNv4 (arXiv 2401.06197) và mọi config FlashInternImage
    # chính thức: kernel CUDA được thiết kế quanh channels_per_group = 16 cố định
    # (vd InternImage-L: channels=160, groups=[10,20,40,80] -> luôn ra D=16).
    # KHÔNG dùng D nhỏ hơn (4, 8) -- đó là nguyên nhân AssertionError trước đó.
    D_PER_GROUP = 16
    assert C % D_PER_GROUP == 0, f"channels={C} phải chia hết cho {D_PER_GROUP}"
    group = C // D_PER_GROUP
    print(f"channels={C}, group={group}, channels_per_group={C // group}")

    module = DCNv4(channels=C, kernel_size=3, stride=1, group=group).to(device)
    x = torch.randn(N, H * W, C, device=device, requires_grad=True)

    out = module(x, shape=(N, H, W, C))
    print("forward OK, output shape:", tuple(out.shape))

    loss = out.sum()
    loss.backward()
    print("backward OK, grad norm:", float(x.grad.norm()))

    # Test riêng ở resolution S2 thật (512x512, channels=32) để đo tốc độ
    import time

    N, H, W, C = 2, 512, 512, 32
    module2 = DCNv4(channels=C, kernel_size=5, stride=1, group=C // D_PER_GROUP).to(device)
    x2 = torch.randn(N, H * W, C, device=device, requires_grad=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        out2 = module2(x2, shape=(N, H, W, C))
        out2.sum().backward()
        x2.grad = None
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 10
    print(f"\nS2 (512x512, C=32): {elapsed*1000:.1f} ms/iter (forward+backward)")

    print("\n>>> TẤT CẢ TEST PASS. An toàn để tích hợp vào decoder.py.")
except Exception as error:  # noqa: BLE001
    import traceback

    print(f"\n>>> LỖI: {type(error).__name__}: {error}")
    traceback.print_exc()
    print(">>> DỪNG Ở ĐÂY. Không nên tích hợp DCNv4 vào training pipeline.")
    sys.exit(1)
