"""Untie an original shared transform into exact per-king-square copies."""
import hashlib
import os
import pathlib
import torch
from train import Net, INPUTS


def expand_checkpoint(source, destination):
    source, destination = pathlib.Path(source), pathlib.Path(destination)
    saved = torch.load(source, map_location="cpu", weights_only=False)
    config = saved["configuration"]
    if config["king_buckets"] != 1 or saved.get("partial_epoch", False):
        raise ValueError("expansion requires a complete shared-context checkpoint")
    original = Net(config["hidden"], config["buckets"], 1)
    original.load_state_dict(saved["model"])
    parameters = list(original.named_parameters())
    groups = saved["optimizer"]["param_groups"]
    if len(groups) != 1 or len(groups[0]["params"]) != len(parameters):
        raise ValueError("unexpected optimizer parameter layout")
    ft_id = None
    for (name, parameter), parameter_id in zip(parameters, groups[0]["params"]):
        if not torch.isfinite(saved["model"][name]).all():
            raise ValueError("nonfinite model state")
        state = saved["optimizer"]["state"][parameter_id]
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state:
                if state[key].shape != parameter.shape or not torch.isfinite(state[key]).all():
                    raise ValueError("invalid optimizer moments")
        if name == "ft.weight":
            ft_id = parameter_id
    assert ft_id is not None and saved["model"]["ft.weight"].shape == (INPUTS, config["hidden"])
    saved["model"]["ft.weight"] = saved["model"]["ft.weight"].repeat(64, 1)
    state = saved["optimizer"]["state"][ft_id]
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        if key in state:
            state[key] = state[key].repeat(64, 1)
    config["king_buckets"] = 64
    saved["context_expansion"] = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_epoch": saved["epoch"], "from_contexts": 1, "to_contexts": 64,
        "policy": "clone feature weights and Adam moments; preserve all other tensors, step counters, scheduler, RNG, data count and epoch numbering",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = str(destination) + ".tmp"
    torch.save(saved, temporary)
    os.replace(temporary, destination)
    return saved["context_expansion"]
