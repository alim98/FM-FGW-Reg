#!/usr/bin/env python3
"""
تست سریع FM-FGW-Reg
این اسکریپت میسازه دو تا volume synthetic و registration اجرا میکنه
"""

import numpy as np
import sys

print("="*60)
print("تست سریع FM-FGW-Reg")
print("="*60)

# Step 1: چک کردن import ها
print("\n[1/6] چک کردن کتابخونه‌ها...")
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except Exception as e:
    print(f"❌ PyTorch: {e}")
    sys.exit(1)

try:
    import ot
    print(f"✅ POT")
except Exception as e:
    print(f"❌ POT: {e}")
    sys.exit(1)

try:
    from fmfgwreg.core import FMFGWReg, RegistrationConfig
    print(f"✅ FM-FGW-Reg")
except Exception as e:
    print(f"❌ FM-FGW-Reg import failed: {e}")
    sys.exit(1)

# Step 2: ساخت دیتای synthetic
print("\n[2/6] ساخت دیتای synthetic...")
def create_sphere(shape=(64, 64, 32), center=None, radius=10):
    """یه کره توی 3D میسازه"""
    if center is None:
        center = np.array(shape) // 2
    
    volume = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                if dist < radius:
                    volume[i,j,k] = 1.0
    
    # اضافه کردن کمی noise
    volume += np.random.randn(*shape) * 0.05
    return volume

# Fixed volume
fixed = create_sphere((64, 64, 32), center=(32, 32, 16), radius=8)
print(f"   Fixed shape: {fixed.shape}")

# Moving volume (shifted)
moving = create_sphere((64, 64, 32), center=(35, 35, 18), radius=8)
print(f"   Moving shape: {moving.shape}")

# Step 3: تنظیمات
print("\n[3/6] تنظیم config...")
config = RegistrationConfig()
config.feature.device = 'cpu'  # برای تست از CPU استفاده میکنیم
config.graph.num_nodes = 100  # تعداد کم برای سرعت
config.fgw.max_iter = 20
config.verbose = False
print(f"   Nodes: {config.graph.num_nodes}")
print(f"   FGW iterations: {config.fgw.max_iter}")

# Step 4: ساخت registration object
print("\n[4/6] ساخت FM-FGW-Reg...")
try:
    reg = FMFGWReg(config)
    print("✅ Registration object ساخته شد")
except Exception as e:
    print(f"❌ خطا: {e}")
    sys.exit(1)

# Step 5: اجرای registration
print("\n[5/6] اجرای registration...")
spacing = (1.0, 1.0, 1.0)

try:
    result = reg.register(
        fixed, moving,
        spacing, spacing,
        do_rigid_prealign=False,  # بدون rigid برای سرعت
    )
    print("✅ Registration تموم شد")
except Exception as e:
    print(f"❌ خطا در registration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: چک کردن نتایج
print("\n[6/6] چک کردن نتایج...")
warped = result['warped']
dvf = result['dvf']

print(f"   Warped shape: {warped.shape}")
print(f"   DVF shape: {dvf.shape}")

# محاسبه MSE قبل و بعد
mse_before = np.mean((fixed - moving)**2)
mse_after = np.mean((fixed - warped)**2)
improvement = (mse_before - mse_after) / mse_before * 100

print(f"\n📊 نتایج:")
print(f"   MSE قبل: {mse_before:.6f}")
print(f"   MSE بعد: {mse_after:.6f}")
print(f"   بهبود: {improvement:.1f}%")

# چک DVF
dvf_mean_magnitude = np.mean(np.linalg.norm(dvf, axis=-1))
print(f"   میانگین displacement: {dvf_mean_magnitude:.2f} voxels")

# Timing
print(f"\n⏱️  زمان کل: {result['timing']['total']:.2f}s")

# چک کردن باگ RBF
if dvf_mean_magnitude < 0.1:
    print("\n⚠️  هشدار: DVF تقریباً صفر است!")
    print("   ممکنه باگ RBF kernel باشه")
else:
    print("\n✅ DVF معقول به نظر میرسه")

if improvement > 0:
    print("\n✅ ✅ ✅ همه چی خوبه! کد کار میکنه!")
else:
    print("\n⚠️  بهبود منفی - ممکنه مشکلی باشه")

print("\n" + "="*60)
print("تست تموم شد")
print("="*60)

