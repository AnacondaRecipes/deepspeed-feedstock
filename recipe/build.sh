#!/bin/bash
set -ex

# Drop the PyPI `ninja` requirement from the wheel metadata: the conda run dep is
# `ninja-base` (the executable runtime JIT needs), which doesn't provide the `ninja`
# distribution, so `pip check` would report it missing. Name translation, not removal.
sed -i.bak 's@^ninja$@#ninja@g' requirements/requirements.txt

if [[ ${cuda_compiler_version} != "None" ]]; then
  # Precompile all DeepSpeed ops on CUDA builds (matches conda-forge).
  export DS_BUILD_OPS=1

  # Compute capabilities match pytorch-feedstock's TORCH_CUDA_ARCH_LIST per platform:
  # kernels for an arch pytorch doesn't ship can't run, since torch itself can't.
  case ${cuda_compiler_version} in
    12.9)
      # sm_60 Pascal and sm_70 Volta dropped, as in pytorch's CUDA 12.x build.
      export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX"
      ;;
    13.0)
      if [[ ${target_platform} == linux-aarch64 ]]; then
        # pytorch's aarch64 CUDA 13 list: no 7.5/8.6 (x86_64-only SKUs), adds
        # 11.0 (Jetson Thor) and 12.1 (DGX Spark).
        export TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0;11.0;12.0;12.1+PTX"
      else
        # CUDA 13.0 drops <7.5 (Volta and earlier).
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX"
      fi
      ;;
    *)
      echo "Unhandled cuda_compiler_version=${cuda_compiler_version}; update build.sh."
      exit 1
      ;;
  esac

  # oneCCL is Intel x86_64-only; skip the CCL comm op on aarch64.
  if [[ ${target_platform} == linux-aarch64 ]]; then
    export DS_BUILD_CCL_COMM=0
  fi
else
  # CPU build: rely on JIT compilation at runtime for any op the user invokes.
  export DS_BUILD_OPS=0
fi

# sparse_attn pins triton==1.0.0, which we don't ship.
export DS_BUILD_SPARSE_ATTN=0

# cutlass_ops and ragged_device_ops require the closed-source `dskernels` Python
# package (`deepspeed-kernels` on PyPI, Linux x86_64-only) at build time — its
# `is_compatible()` doesn't check for it, but `extra_ldflags()` does. Disable
# them to keep the build self-contained. Users who want DeepSpeed Inference v2
# kernels can `pip install deepspeed-kernels` and JIT-compile at runtime.
export DS_BUILD_CUTLASS_OPS=0
export DS_BUILD_RAGGED_DEVICE_OPS=0

${PYTHON} -m pip install . --no-deps --no-build-isolation -vv
