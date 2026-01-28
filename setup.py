# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Setup script for ppisp CUDA extension.
Based on https://github.com/rahul-goel/fused-ssim/blob/main/setup.py
"""

import os
import re
import sys

from setuptools import setup


def get_project_metadata():
    """
    Extract name/version from pyproject.toml to prevent older setuptools or
    non-isolated builds from falling back to UNKNOWN.

    - Python >= 3.11: Prefer the stdlib TOML parser (tomllib)
    - Python 3.10: Fall back to minimal regex parsing (only [project] name/version)
    """
    project_file = os.path.join(os.path.dirname(__file__), "pyproject.toml")
    if not os.path.exists(project_file):
        raise FileNotFoundError("pyproject.toml not found")

    with open(project_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Prefer stdlib TOML parser when available.
    try:
        import tomllib  # Python 3.11+
    except Exception:  # pragma: no cover
        tomllib = None

    if tomllib is not None:
        data = tomllib.loads(content)
        project = data.get("project", {}) if isinstance(data, dict) else {}
        name = project.get("name")
        version = project.get("version")
        if isinstance(name, str) and isinstance(version, str) and name and version:
            return {"name": name, "version": version}

    # Minimal regex fallback (compatible with Python 3.10, no third-party deps).
    project_section_match = re.search(
        r"(?ms)^\[project\]\s*$\n(.*?)(?:^\[.*?\]\s*$|\Z)", content
    )
    project_section = project_section_match.group(1) if project_section_match else content
    name_match = re.search(
        r'(?m)^\s*name\s*=\s*["\']([^"\']+)["\']\s*$', project_section
    )
    version_match = re.search(
        r'(?m)^\s*version\s*=\s*["\']([^"\']+)["\']\s*$', project_section
    )

    name = name_match.group(1) if name_match else None
    version = version_match.group(1) if version_match else None
    if not name or not version:
        raise RuntimeError("Could not extract project name/version from pyproject.toml")

    return {"name": name, "version": version}


metadata = get_project_metadata()


# Optional: fail fast with a clear error message when building from source without PyTorch installed.
try:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
except ImportError:
    print("Error: PyTorch must be installed before installing/building this package.")
    print("Try running: pip install torch")
    raise

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stderr.reconfigure(line_buffering=True)


# Default fallback architectures
fallback_archs = [
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_120,code=sm_120",
]

nvcc_args = [
    "-O3",
    "--use_fast_math",
]
nvcc_args += ["-lineinfo", "--generate-line-info", "--source-in-ptx"]

detected_arch = None

if torch.cuda.is_available():
    try:
        device = torch.cuda.current_device()
        compute_capability = torch.cuda.get_device_capability(device)
        arch = f"sm_{compute_capability[0]}{compute_capability[1]}"

        arch_msg = f"Detected GPU architecture: {arch}"
        print(arch_msg)
        print(arch_msg, file=sys.stderr, flush=True)

        nvcc_args.append(f"-arch={arch}")
        detected_arch = arch
    except Exception as e:
        error_msg = f"Failed to detect GPU architecture: {e}. Falling back to multiple architectures."
        print(error_msg)
        print(error_msg, file=sys.stderr, flush=True)
        nvcc_args.extend(fallback_archs)
else:
    cuda_msg = "CUDA not available. Falling back to multiple architectures."
    print(cuda_msg)
    print(cuda_msg, file=sys.stderr, flush=True)
    nvcc_args.extend(fallback_archs)


class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        arch_info = f"Building with GPU architecture: {detected_arch if detected_arch else 'multiple architectures'}"
        print("\n" + "=" * 50)
        print(arch_info)
        print("=" * 50 + "\n")
        super().build_extensions()


setup(
    name=metadata["name"],
    version=metadata["version"],
    packages=["ppisp"],
    ext_modules=[
        CUDAExtension(
            name="ppisp_cuda",
            sources=[
                "ppisp/src/ppisp_impl.cu",
                "ppisp/ext.cpp"
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": nvcc_args
            }
        )
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    }
)

final_msg = f"Setup completed. NVCC args: {nvcc_args}"
print(final_msg)
