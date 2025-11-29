# from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
# from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
# from xtuner.v1.model import Qwen3VLDense4BConfig
# sample_max_length = 8192
# model_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/models/Qwen3-VL-4B-Instruct"
# model_cfg = Qwen3VLDense4BConfig(freeze_vision=True, freeze_projector=True)
# oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": "~/petreloss.conf"})
# _data = {}
# config = Qwen3VLTokenizeFnConfig(
#             model_config=model_cfg,
#             max_length=sample_max_length,
#             min_pixels=_data.get('image_min_pixels', 28*28),
#             max_pixels=_data.get('image_max_pixels', int(sample_max_length * 0.8 * 28 * 28)),
#             video_max_total_pixels=_data.get('video_max_total_pixels', int(sample_max_length * 0.8 * 4 * 28 * 28)),
#             video_min_frames=_data.get('video_min_frames', 1),
#             video_max_frames=_data.get('video_max_frames', 2048),
#             fps=_data.get('video_sample_fps', 4),
#             processor_path=model_path,
#             system_message=_data.get('system_message', None),
#             hash=_data.get('hash', None),
#             oss_loader_cfg=oss_loader_cfg
#         )

from transformers import AutoConfig
from xtuner.v1.utils import is_hf_model_path
path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/work_dir/VideoChat3_4B_train_stage1-1/20251127192016/hf-latest"
# AutoConfig.from_pretrained(path, trust_remote_code=True)
print(is_hf_model_path(path))