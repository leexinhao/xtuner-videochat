
from safetensors import safe_open

tensor_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/VideoChat3-2B/model.safetensors"


print(safe_open(tensor_path, framework="pt").keys())