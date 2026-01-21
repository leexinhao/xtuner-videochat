import os
from packaging import version
import parametrize
import torch
from xtuner._testing import patch_hf_rms_norm, DeterministicDDPTestCase
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
import tempfile
from pathlib import Path
import json
from safetensors import safe_open
from unittest import skipIf
import transformers
from xtuner.v1.model.compose.videochat3.videochat3_config import VideoChat3Dense2BConfig
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContextInputItem
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.config import FSDPConfig
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.utils.test_utils import init_data_mesh
from pathlib import Path

origin_hf_path = Path("/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/VideoChat3-2B")
origin_index_path = origin_hf_path / "model.safetensors.index.json"
tmpdir = Path("/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/VideoChat3-2B-savehf-debug")
saved_index_path = tmpdir / "model.safetensors.index.json"

cache_save_fh = {}

with open(origin_index_path, "r") as f:
    origin_index = json.load(f)
with open(saved_index_path, "r") as f:
    saved_index = json.load(f)

for key in origin_index["weight_map"].keys():
    origin_safetensor_name = origin_index["weight_map"][key]
    saved_safetensor_name = saved_index["weight_map"][key]

    origin_sf_fh_name = str(origin_hf_path / origin_safetensor_name)
    expected_sf_fh_name = str(tmpdir / saved_safetensor_name)

    if origin_safetensor_name not in cache_save_fh:
        cache_save_fh[origin_safetensor_name] = safe_open(origin_sf_fh_name, framework="pt")
    if saved_safetensor_name not in cache_save_fh:
        cache_save_fh[saved_safetensor_name] = safe_open(expected_sf_fh_name, framework="pt")

    origin_fh = cache_save_fh[origin_safetensor_name]
    saved_fh = cache_save_fh[saved_safetensor_name]

    origin_tensor = origin_fh.get_tensor(key)
    saved_tensor = saved_fh.get_tensor(key)
    if not torch.equal(origin_tensor, saved_tensor):
        print(key)
        print(torch.equal(origin_tensor.to(saved_tensor.dtype), saved_tensor))
        print(origin_tensor.dtype)
        print("//////////////////////////////////////////////")
        print(saved_tensor.dtype)


# 测试 safetensors 中的张量数量是否与模型索引中的张量数量匹配
safetensor_keys = []
for safetensor_path in tmpdir.glob("*.safetensors"):
    fh = cache_save_fh[safetensor_path.name]
    safetensor_keys.extend(fh.keys())

safetensor_keys.sort()

model_index_keys = list(saved_index["weight_map"].keys())
model_index_keys.sort()

if safetensor_keys == model_index_keys:
    print("两个列表内容完全相同")
else:
    print("两个列表内容不相同")
        