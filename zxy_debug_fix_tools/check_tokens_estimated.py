import json
import random
import os
from transformers import AutoTokenizer
from xtuner.v1.datasets.mllm_tokenize_fn.videochat3_tokenize_fn import VideoChat3TokenizeFnConfig

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def main():
    model_path = "/mnt/petrelfs/zengxiangyu/Research_lixinhao/xtuner-videochat/work_dir/stage3/VideoChat3_4B_train_stage3_minisft_v21_lr8e-5_vtlr2e-5_sohav2_long_v5/20260331111908/hf-latest"
    meta_data_path = 'training_data_annotations/stage4_online/Online_Subset/Online_StreamingQA_120K.json'
    sample_max_length = 16384 * 4

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    with open(meta_data_path, 'r') as f:
        ds_collections = json.load(f)

    mismatched_jsonls = set()

    for name, _data in ds_collections.items():
        anno_path = _data['anno_path']
        media_root = _data.get('media_root', '')
        
        if not os.path.exists(anno_path):
            print(f"Skipping {name}, file not found: {anno_path}")
            continue
            
        print(f"\nProcessing {name}...")
        
        # Load a few random lines
        with open(anno_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            continue
            
        sampled_lines = random.sample(lines, min(50, len(lines)))
        
        # Initialize tokenize function
        tokenize_cfg = VideoChat3TokenizeFnConfig(
            max_length=sample_max_length,
            image_min_pixels=_data.get('image_min_pixels', 28*28),
            image_max_pixels=_data.get('image_max_pixels', int(sample_max_length * 0.8 * 28 * 28)),
            frame_min_pixels=_data.get('frame_min_pixels', 28*28),
            frame_max_pixels=_data.get('frame_max_pixels', 448*448),
            video_max_total_pixels=_data.get('video_max_total_pixels', int(sample_max_length * 0.8 * 4 * 28 * 28)),
            video_min_frames=_data.get('video_min_frames', 1),
            video_max_frames=_data.get('video_max_frames', 3600), 
            fixed_num_sampled_frames=_data.get('fixed_num_sampled_frames', None),
            video_sample_fps=_data.get('video_sample_fps', 2), 
            processor_path=model_path,
            system_message=_data.get('system_message', None),
            hash=_data.get('hash', None),
        )
        
        tokenize_fn = tokenize_cfg.build(tokenizer, anno_name=name)
        
        for line in sampled_lines:
            try:
                item = json.loads(line)
                
                # Get estimated tokens (cache mode)
                tokenize_fn.state = "cache"
                cache_res = tokenize_fn(item, media_root=media_root)
                estimated_tokens = cache_res.get("num_tokens", 0)
                
                # Get actual tokens (normal mode)
                tokenize_fn.state = "normal"
                actual_res = tokenize_fn(item, media_root=media_root)
                actual_tokens = actual_res.get('num_tokens', 0) if isinstance(actual_res, dict) else getattr(actual_res, 'num_tokens', 0)
                
                if estimated_tokens == actual_tokens:
                    print(f"{GREEN}MATCH in {name} (ID: {item.get('id', 'unknown')}): Tokens = {actual_tokens}{RESET}")
                else:
                    print(f"{RED}MISMATCH in {name} (ID: {item.get('id', 'unknown')}): Estimated = {estimated_tokens}, Actual = {actual_tokens}{RESET}")
                    mismatched_jsonls.add(anno_path)
                    
                # Free memory
                del cache_res
                del actual_res
                
                # Clear tokenize_fn internal states if any
                tokenize_fn._image_path = []
                tokenize_fn._video_path = []
                tokenize_fn._image_wh_list = []
                tokenize_fn._video_meta_list = []
                
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing item in {name}: {e}")

    print("\n" + "="*80)
    if mismatched_jsonls:
        print(f"{RED}Found mismatches in the following JSONL files:{RESET}")
        for path in mismatched_jsonls:
            print(f"{RED}- {path}{RESET}")
    else:
        print(f"{GREEN}All sampled data matched perfectly!{RESET}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()