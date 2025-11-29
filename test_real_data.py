#!/usr/bin/env python3
"""
تست کامل FM-FGW-Reg با دیتای واقعی IXI
"""

import numpy as np
import sys
from pathlib import Path
import time

print("="*70)
print("تست FM-FGW-Reg با دیتای واقعی (IXI Brain MRI)")
print("="*70)

# Step 1: Load data
print("\n[1/5] بارگذاری دیتا...")
try:
    import nibabel as nib
    
    data_dir = Path("/u/almik/REG/data/ixi_eval")
    files = sorted(list(data_dir.glob("*.nii.gz")))[:2]  # فقط 2 تا اول
    
    if len(files) < 2:
        print("❌ کمتر از 2 فایل پیدا شد!")
        sys.exit(1)
    
    print(f"   Fixed: {files[0].name}")
    print(f"   Moving: {files[1].name}")
    
    # Load volumes
    fixed_nii = nib.load(str(files[0]))
    moving_nii = nib.load(str(files[1]))
    
    fixed = fixed_nii.get_fdata()
    moving = moving_nii.get_fdata()
    
    # Get spacing from affine
    fixed_spacing = tuple(np.abs(np.diag(fixed_nii.affine)[:3]))
    moving_spacing = tuple(np.abs(np.diag(moving_nii.affine)[:3]))
    
    print(f"   Fixed shape: {fixed.shape}, spacing: {fixed_spacing}")
    print(f"   Moving shape: {moving.shape}, spacing: {moving_spacing}")
    
except Exception as e:
    print(f"❌ خطا در بارگذاری: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Setup config
print("\n[2/5] تنظیم config...")
try:
    from fmfgwreg import FMFGWReg, RegistrationConfig
    
    config = RegistrationConfig()
    config.feature.device = 'cpu'  # برای اینکه همه بتونن اجرا کنن
    config.graph.num_nodes = 200   # تعداد کم برای سرعت
    config.fgw.max_iter = 30
    config.rigid_prealign = True   # استفاده از rigid
    config.use_cache = True
    config.verbose = True
    
    print(f"   Device: {config.feature.device}")
    print(f"   Nodes: {config.graph.num_nodes}")
    print(f"   Rigid prealign: {config.rigid_prealign}")
    print(f"   Cache: {config.use_cache}")
    
except Exception as e:
    print(f"❌ خطا در config: {e}")
    sys.exit(1)

# Step 3: Create registration object
print("\n[3/5] ساخت registration object...")
start_time = time.time()
try:
    reg = FMFGWReg(config)
    print(f"   ✅ ساخته شد ({time.time()-start_time:.1f}s)")
except Exception as e:
    print(f"❌ خطا: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Run registration
print("\n[4/5] اجرای registration...")
print("   ⏳ این ممکنه چند دقیقه طول بکشه...")
reg_start = time.time()

try:
    result = reg.register(
        fixed, moving,
        fixed_spacing, moving_spacing,
        fixed_id=files[0].stem,
        moving_id=files[1].stem,
        do_rigid_prealign=config.rigid_prealign,
    )
    
    reg_time = time.time() - reg_start
    print(f"   ✅ تموم شد ({reg_time:.1f}s)")
    
except Exception as e:
    print(f"❌ خطا در registration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Analyze results
print("\n[5/5] تحلیل نتایج...")
try:
    warped = result['warped']
    dvf = result['dvf']
    
    print(f"\n📊 Shapes:")
    print(f"   Fixed: {fixed.shape}")
    print(f"   Moving: {moving.shape}")
    print(f"   Warped: {warped.shape}")
    print(f"   DVF: {dvf.shape}")
    
    # Compute metrics
    mse_before = np.mean((fixed - moving)**2)
    mse_after = np.mean((fixed - warped)**2)
    improvement = (mse_before - mse_after) / mse_before * 100
    
    # DVF statistics
    dvf_magnitude = np.linalg.norm(dvf, axis=-1)
    dvf_mean = np.mean(dvf_magnitude)
    dvf_max = np.max(dvf_magnitude)
    dvf_std = np.std(dvf_magnitude)
    
    print(f"\n📈 Registration Quality:")
    print(f"   MSE قبل: {mse_before:.2f}")
    print(f"   MSE بعد: {mse_after:.2f}")
    print(f"   بهبود: {improvement:.1f}%")
    
    print(f"\n🎯 DVF Statistics:")
    print(f"   Mean magnitude: {dvf_mean:.2f} voxels")
    print(f"   Max magnitude: {dvf_max:.2f} voxels")
    print(f"   Std: {dvf_std:.2f} voxels")
    
    # Timing breakdown
    if 'timing' in result:
        print(f"\n⏱️  Timing:")
        for key, val in result['timing'].items():
            print(f"   {key}: {val:.2f}s")
    
    # Quality checks
    print(f"\n✅ Quality Checks:")
    
    if dvf_mean > 0.5:
        print(f"   ✅ DVF non-zero ({dvf_mean:.2f} voxels)")
    else:
        print(f"   ⚠️  DVF خیلی کوچیکه ({dvf_mean:.2f} voxels)")
    
    if improvement > 0:
        print(f"   ✅ MSE بهبود یافته ({improvement:.1f}%)")
    else:
        print(f"   ⚠️  MSE بدتر شده ({improvement:.1f}%)")
    
    if 'coupling' in result:
        T = result['coupling']
        coupling_sparsity = (T < 0.001).sum() / T.size * 100
        print(f"   Coupling sparsity: {coupling_sparsity:.1f}%")
    
    # Save results
    output_dir = Path("/u/almik/REG/test_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 ذخیره نتایج در {output_dir}...")
    
    # Save warped
    warped_nii = nib.Nifti1Image(warped.astype(np.float32), fixed_nii.affine)
    nib.save(warped_nii, output_dir / "warped.nii.gz")
    print(f"   ✅ warped.nii.gz")
    
    # Save DVF
    dvf_nii = nib.Nifti1Image(dvf.astype(np.float32), fixed_nii.affine)
    nib.save(dvf_nii, output_dir / "dvf.nii.gz")
    print(f"   ✅ dvf.nii.gz")
    
    print("\n" + "="*70)
    print("🎉 تست موفقیت‌آمیز بود!")
    print("="*70)
    print(f"\nنتایج در: {output_dir}")
    print(f"زمان کل: {time.time() - start_time:.1f}s")
    
except Exception as e:
    print(f"❌ خطا در تحلیل: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

