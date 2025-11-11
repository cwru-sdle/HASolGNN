#!/usr/bin/env python3
"""
codes/infer.py

One-command inference for HASolGNN or HASolGNN-LLMs.
Run example:
  python codes/infer.py \
    --checkpoint checkpoints/best_ckpt.pt \
    --solute "CCO" \
    --solvent "O" \
    --temp 298.15 \
    --temp-units K \
    --device cpu \
    --default-y-mean -0.842 \
    --default-y-std 0.987
"""

import argparse
import importlib
import json
import os
import sys
from typing import Tuple, Dict

import torch
import numpy as np

# optional: RDKit for SMILES -> graph conversion
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
except Exception:
    Chem = None


# ------------------------------
# Utility functions
# ------------------------------

def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt


def try_repo_featurizer(sol_smiles: str, solv_smiles: str, temp: float):
    """
    Try to import a featurizer implemented in the repo (common names).
    If none available, return None so fallback is used.
    """
    candidates = [
        "codes.featurize.build_interaction_graph",
        "codes.data.build_graph_from_smiles",
        "codes.utils.featurize_pair",
    ]
    for modpath in candidates:
        try:
            mod_name, fn_name = modpath.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name)
            print(f"[INFO] Using repository featurizer: {modpath}")
            return fn(sol_smiles, solv_smiles, temp)
        except Exception:
            continue
    return None


def fallback_simple_features(sol_smiles: str, solv_smiles: str, temp: float):
    """
    Minimal fallback: compute Morgan fingerprints for solute & solvent,
    concatenate with temperature, return as tensor input.
    """
    if Chem is None:
        raise RuntimeError("RDKit not available. Install RDKit or provide repository featurizer.")

    def fp_vec(smiles, radius=2, nbits=2048):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
        arr = np.zeros((nbits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr

    sol_fp = fp_vec(sol_smiles)
    solv_fp = fp_vec(solv_smiles)
    feat = np.concatenate([sol_fp, solv_fp, np.array([temp], dtype=np.float32)])
    return torch.tensor(feat[None, :], dtype=torch.float32)  # batch dim


def build_model_from_checkpoint(ckpt: dict, device: torch.device):
    """
    Build model from checkpoint in various formats.
    """
    # Case A: full model object
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], torch.nn.Module):
        model = ckpt["model"].to(device)
        return model

    # Case B: model_state_dict + config
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        cfg = ckpt.get("config", {})
        model_path = cfg.get("model_class") or cfg.get("model_import_path")
        if model_path:
            try:
                mod_path, cls_name = model_path.rsplit(".", 1)
                mod = importlib.import_module(mod_path)
                ModelClass = getattr(mod, cls_name)
                model = ModelClass(**cfg.get("model_hparams", {}))
                model.load_state_dict(ckpt["model_state_dict"])
                model.to(device)
                print(f"[INFO] Loaded model from class {model_path}")
                return model
            except Exception as e:
                print("[WARN] Could not import model class:", e)

    # Case C: TorchScript model
    try:
        model = torch.jit.load(args.checkpoint, map_location=device)
        model.to(device)
        print("[INFO] Loaded TorchScript model.")
        return model
    except Exception:
        pass

    # Case D: generic MLP fallback
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", None))
    if state is not None:
        first_key = list(state.keys())[0]
        in_dim = state[first_key].shape[1] if len(state[first_key].shape) > 1 else 512
        model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        model.load_state_dict(state, strict=False)
        model.to(device)
        print(f"[WARN] Using generic MLP fallback (input_dim={in_dim})")
        return model

    raise RuntimeError("Could not reconstruct model from checkpoint.")


# ------------------------------
# Main inference function
# ------------------------------

def main(args):
    device = torch.device(args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu")

    # load checkpoint
    ckpt = load_checkpoint(args.checkpoint)

    # read normalization constants
    target_mean = ckpt.get("target_mean") or ckpt.get("y_mean")
    target_std = ckpt.get("target_std") or ckpt.get("y_std")
    scaler = ckpt.get("scaler", None)
    if scaler and isinstance(scaler, dict):
        target_mean = target_mean or scaler.get("mean")
        target_std = target_std or scaler.get("std")

    if target_mean is None or target_std is None:
        target_mean = float(args.default_y_mean)
        target_std = float(args.default_y_std)
        print(f"[WARN] Normalization constants missing — using defaults mean={target_mean}, std={target_std}")

    # prepare input features
    feats = try_repo_featurizer(args.solute, args.solvent, float(args.temp))
    if feats is None:
        feats = fallback_simple_features(args.solute, args.solvent, float(args.temp))

    # build model
    model = build_model_from_checkpoint(ckpt, device)
    model.eval()

    with torch.no_grad():
        feats = feats.to(device)
        out = model(feats)

        # extract tensor value
        if isinstance(out, dict):
            raw = out.get("pred") or out.get("y_hat") or list(out.values())[0]
        else:
            raw = out

        raw_val = raw.detach().cpu().numpy().squeeze()
        pred_logS = raw_val * target_std + target_mean

        print("\n=== HASolGNN Inference ===")
        print(f"Solute SMILES  : {args.solute}")
        print(f"Solvent SMILES : {args.solvent}")
        print(f"Temperature    : {args.temp} ({args.temp_units})")
        print(f"Pred (normalized) : {raw_val:.6f}")
        print(f"Pred logS         : {pred_logS:.6f}")
        print("===========================\n")


# ------------------------------
# Entry point
# ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pt or .pth)")
    parser.add_argument("--solute", required=True, help="Solute SMILES (quoted)")
    parser.add_argument("--solvent", required=True, help="Solvent SMILES (quoted)")
    parser.add_argument("--temp", required=True, help="Temperature value (numeric)")
    parser.add_argument("--temp-units", default="K", choices=["K", "C"], help="Temperature units (K or C)")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda:0")
    parser.add_argument("--default-y-mean", default="0.0", help="Fallback target mean if not in checkpoint")
    parser.add_argument("--default-y-std", default="1.0", help="Fallback target std if not in checkpoint")
    args = parser.parse_args()
    main(args)
