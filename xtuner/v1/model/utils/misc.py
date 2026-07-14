from typing import Any

import torch
import torch.distributed as dist


def module_dict_repr(self):
    """Return a custom repr for ModuleList that compresses repeated module
    representations."""

    def _addindent(s_, numSpaces):
        s = s_.split("\n")
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * " ") + line for line in s]
        s = "\n".join(s)
        s = first + "\n" + s
        return s

    list_of_reprs = [repr(item) for item in self.values()]
    if len(list_of_reprs) == 0:
        return self._get_name() + "()"

    start_end_indices = [[0, 0]]
    repeated_blocks = [list_of_reprs[0]]
    for i, r in enumerate(list_of_reprs[1:], 1):
        if r == repeated_blocks[-1]:
            start_end_indices[-1][1] += 1
            continue

        start_end_indices.append([i, i])
        repeated_blocks.append(r)

    lines = []
    main_str = self._get_name() + "("
    for (start_id, end_id), b in zip(start_end_indices, repeated_blocks):
        local_repr = f"({start_id}): {b}"  # default repr

        if start_id != end_id:
            n = end_id - start_id + 1
            local_repr = f"({start_id}-{end_id}): {n} x {b}"

        local_repr = _addindent(local_repr, 2)
        lines.append(local_repr)

    main_str += "\n  " + "\n  ".join(lines) + "\n"
    main_str += ")"
    return main_str


class ModelForwardExtraLogInfo(dict):
    """An extensible dictionary for carrying extra information in the model's
    output.

    In the Reinforcement Learning (RL) training process, this information will be processed by the `TrainingWorker`.
    In the SFT/Pretraining process, this information will be processed by the `Trainer`.
    """

    # Tensor to store the maximum model params update ratio.
    # Shape: `(n_chunk, intra_layer_micro_batch, 1)` if intra_layer_micro_batch > 1 else `(n_chunk, 1)`
    max_ratio: torch.Tensor
    # Tensor to store the ranking loss for logging.
    # Shape: `(intra_layer_micro_batch, 1)` if intra_layer_micro_batch > 1 else `(1,)`
    log_rank_loss: torch.Tensor

    def __init__(self, init_dict: dict[str, Any] = {}):
        super().__init__()
        if init_dict:
            for k, v in init_dict.items():
                self[k] = v

    def append(self, input_info: dict[str, Any]):
        for key, tensor in input_info.items():
            if key in self and isinstance(self[key], list):
                self[key].append(tensor)
                continue

            # 统一处理为 2D 张量后在第 0 维拼接
            tensor_view = tensor
            if tensor_view.dim() == 0:
                tensor_view = tensor_view.unsqueeze(0)
            tensor_view = tensor_view.unsqueeze(0)

            if key in self:
                if self[key].shape[1:] != tensor_view.shape[1:]:
                    # Dimensions mismatch (variable length), fallback to list
                    self[key] = [self[key], tensor]
                else:
                    self[key] = torch.cat([self[key], tensor_view], dim=0)
            else:
                self[key] = tensor_view

    def get(self):
        return_dict = {}
        # 当增加新的字段时，需要在这里添加相应的处理逻辑
        if "max_ratio" in self:
            while self["max_ratio"].dim() >= 1:
                self["max_ratio"] = torch.max(self["max_ratio"], dim=-1).values
            max_ratio_value = self["max_ratio"].item()
            return_dict["max_ratio"] = max_ratio_value
        if "log_rank_loss" in self:
            while self["log_rank_loss"].dim() >= 1:
                self["log_rank_loss"] = torch.sum(self["log_rank_loss"], dim=-1)
            log_rank_loss_value = self["log_rank_loss"].item()
            return_dict["loss"] = log_rank_loss_value

        # Handle dataset loss with global alignment (avoid all_reduce shape mismatch hang)
        if "_dataset_unique_ids" in self:
            all_unique_ids = self["_dataset_unique_ids"]
            all_loss_sums = self["_dataset_loss_sums"]
            all_weight_sums = self["_dataset_weight_sums"]

            # Handle list (multiple micro-batches) or single tensor
            if isinstance(all_unique_ids, list):
                all_unique_ids = torch.cat([t.view(-1) for t in all_unique_ids], dim=0)
                all_loss_sums = torch.cat([t.view(-1) for t in all_loss_sums], dim=0)
                all_weight_sums = torch.cat([t.view(-1) for t in all_weight_sums], dim=0)
            else:
                all_unique_ids = all_unique_ids.view(-1)
                all_loss_sums = all_loss_sums.view(-1)
                all_weight_sums = all_weight_sums.view(-1)

            # Local aggregate
            unique_ids, inverse_indices = torch.unique(all_unique_ids.to(torch.long), sorted=True, return_inverse=True)
            n_local = unique_ids.size(0)
            local_loss_sums = torch.zeros(n_local, dtype=torch.float32, device=unique_ids.device)
            local_weight_sums = torch.zeros(n_local, dtype=torch.float32, device=unique_ids.device)
            local_loss_sums.scatter_add_(0, inverse_indices, all_loss_sums.float())
            local_weight_sums.scatter_add_(0, inverse_indices, all_weight_sums.float())

            if dist.is_initialized():
                world_size = dist.get_world_size()
                local_len = torch.tensor([n_local], device=unique_ids.device, dtype=torch.long)
                lens = [torch.zeros_like(local_len) for _ in range(world_size)]
                dist.all_gather(lens, local_len)
                max_len = int(torch.max(torch.stack(lens)).item())

                if max_len == 0:
                    global_unique_ids = unique_ids  # empty
                else:
                    pad = torch.full((max_len,), -1, device=unique_ids.device, dtype=torch.long)
                    pad[:n_local] = unique_ids
                    gathered = [torch.empty_like(pad) for _ in range(world_size)]
                    dist.all_gather(gathered, pad)
                    all_ids = torch.cat(gathered, dim=0)
                    all_ids = all_ids[all_ids >= 0]
                    global_unique_ids = torch.unique(all_ids, sorted=True)

                n_global = global_unique_ids.size(0)
                if n_global == 0:
                    final_loss_sums = torch.zeros(0, dtype=torch.float32, device=unique_ids.device)
                    final_weight_sums = torch.zeros(0, dtype=torch.float32, device=unique_ids.device)
                    unique_ids = global_unique_ids
                else:
                    idx = torch.searchsorted(global_unique_ids, unique_ids)
                    final_loss_sums = torch.zeros(n_global, dtype=torch.float32, device=unique_ids.device)
                    final_weight_sums = torch.zeros(n_global, dtype=torch.float32, device=unique_ids.device)
                    final_loss_sums.scatter_add_(0, idx, local_loss_sums)
                    final_weight_sums.scatter_add_(0, idx, local_weight_sums)
                    unique_ids = global_unique_ids

                combined = torch.stack([final_loss_sums, final_weight_sums], dim=0)
                dist.all_reduce(combined, op=dist.ReduceOp.SUM)
                final_loss_sums = combined[0]
                final_weight_sums = combined[1]
            else:
                final_loss_sums = local_loss_sums
                final_weight_sums = local_weight_sums

            unique_ids_cpu = unique_ids.cpu()
            final_loss_sums_cpu = final_loss_sums.cpu()
            final_weight_sums_cpu = final_weight_sums.cpu()

            for i in range(unique_ids_cpu.numel()):
                d_id = unique_ids_cpu[i].item()
                weight = final_weight_sums_cpu[i].item()
                if weight > 0:
                    return_dict[f"dataset_loss_{d_id}"] = final_loss_sums_cpu[i].item() / weight

        return return_dict
