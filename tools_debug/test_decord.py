from decord import VideoReader
from decord import cpu

# 指定视频文件路径
video_path = '/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/tests/resource/tennis.mp4'

# 使用 CPU 作为后端
vr = VideoReader(video_path, ctx=cpu(0))
print(dir(vr))
# 获取第一个帧的尺寸
# vr[0] 实际上会返回一个张量 (Tensor)，其形状为 (H, W, C)
frame_shape = vr[0].shape

# 帧的宽度和高度是尺寸的后两个维度
height = frame_shape[0]
width = frame_shape[1]

print(f"视频帧的高度: {height}")
print(f"视频帧的宽度: {width}")
