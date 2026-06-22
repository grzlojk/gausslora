import argparse
import gc
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    DefaultDataCollator,
    ViTForImageClassification,
    get_cosine_schedule_with_warmup,
)

from adapters import FastGaussianSplattingAdapter, LayerWithAdapter, LoraAdapter
from ds import DatasetManager
from muon import SingleDeviceMuonWithAuxAdam

ADAPTER_MAP = {
    "LoraAdapter": LoraAdapter,
    "FastGaussianSplattingAdapter": FastGaussianSplattingAdapter,
}


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def inject_adapters(model, adapter_classes, config_kwargs):
    adapter_dim = config_kwargs.get("adapter_dim", 32)
    trainable_base = config_kwargs.get("trainable_base", False)
    target_strategy = config_kwargs.get("target_strategy", ["query", "value"])

    target_layers = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        for target in target_strategy:
            if target in name:
                target_layers.append((name, module))
                break

    print(f"   Target strategy: {target_strategy}")
    print(f"   Injecting adapters into {len(target_layers)} layers")

    for full_name, original_layer in target_layers:
        split_name = full_name.rsplit(".", 1)
        parent_module = model.get_submodule(split_name[0]) if len(split_name) > 1 else model
        child_name = split_name[1] if len(split_name) > 1 else full_name

        if isinstance(original_layer, LayerWithAdapter):
            continue

        injection_kwargs = {
            k: v
            for k, v in config_kwargs.items()
            if k not in {"adapter_classes", "trainable_base", "target_strategy", "name", "lr", "muon_learning_rate"}
        }

        wrapped_layer = LayerWithAdapter(
            original_layer,
            adapter_classes,
            adapter_dim=adapter_dim,
            **injection_kwargs,
        )
        wrapped_layer.original_layer.requires_grad_(trainable_base)
        setattr(parent_module, child_name, wrapped_layer)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_labels, exp_config, model_id):
    print(f"   Loading base model: {model_id}")
    model = ViTForImageClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    )

    for param in model.parameters():
        param.requires_grad = False

    adapter_names = exp_config.get("adapter_classes", [])
    adapter_classes = [ADAPTER_MAP[name] for name in adapter_names if name in ADAPTER_MAP]
    if not adapter_classes:
        raise ValueError(f"No supported adapters found in: {adapter_names}")

    inject_adapters(model, adapter_classes, exp_config)

    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, "head"):
        for param in model.head.parameters():
            param.requires_grad = True

    return model


def save_adapter_weights(model, save_dir, run_name):
    os.makedirs(save_dir, exist_ok=True)

    save_data = {
        "generated_matrices": {},
        "adapters_state_dict": {},
        "head_state_dict": {},
    }

    for name, module in model.named_modules():
        if isinstance(module, LayerWithAdapter):
            save_data["generated_matrices"][name] = module.get_adapter_weight()
            save_data["adapters_state_dict"][name] = module.state_dict()

    if hasattr(model, "classifier"):
        save_data["head_state_dict"] = model.classifier.state_dict()
    elif hasattr(model, "head"):
        save_data["head_state_dict"] = model.head.state_dict()

    save_path = os.path.join(save_dir, f"{run_name}_adapters.pt")
    torch.save(save_data, save_path)
    print(f"   Saved adapter weights to: {save_path}")


def run_validation(model, val_loader, device):
    model.eval()
    val_acc = 0.0
    val_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                outputs = model(**batch)
            if hasattr(outputs, "loss"):
                val_loss += outputs.loss.mean().item()
            preds = torch.argmax(outputs.logits, dim=-1)
            val_acc += (preds == batch["labels"]).float().mean().item()
            steps += 1

    return val_acc / steps, val_loss / steps


def train_eval(train_loader, val_loader, train_steps, log_freq, run_name, model, optimizer, device, scaler, grad_accum_steps, config):
    start_time = time.time()
    use_scheduler = config.get("USE_SCHEDULER", True)
    scheduler = None

    if use_scheduler:
        warmup_steps = int(0.1 * train_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=train_steps,
        )
        print(f"   Cosine scheduler active ({train_steps} steps)")
    else:
        print("   Scheduler disabled (constant LR)")

    pbar = tqdm(total=train_steps, desc=f"Train {run_name}", leave=True)
    global_step = 0
    micro_step = 0
    acc_res = []

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)

    while global_step < train_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.amp.autocast("cuda"):
            outputs = model(**batch)
            reg_loss = 0.0
            for module in model.modules():
                if isinstance(module, LayerWithAdapter):
                    reg_loss += module.get_regularization_loss()
            total_loss = (outputs.loss + reg_loss) / grad_accum_steps

        scaler.scale(total_loss).backward()
        micro_step += 1

        if micro_step % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            global_step += 1
            pbar.update(1)

            current_lr = scheduler.get_last_lr()[-1] if scheduler else optimizer.param_groups[-1]["lr"]
            metrics_to_log = {
                "train/loss": total_loss.item() * grad_accum_steps,
                "train/lr": current_lr,
            }

            if global_step > 0 and (global_step % log_freq == 0 or global_step == train_steps):
                print(f"\n   Validating step {global_step}/{train_steps}...")
                val_acc, val_loss = run_validation(model, val_loader, device)
                metrics_to_log.update({"val/accuracy": val_acc, "val/loss": val_loss})
                acc_res.append(val_acc)
                print(f"      Step {global_step} acc: {val_acc:.4f} | loss: {val_loss:.4f}")
                model.train()

            wandb.log(metrics_to_log, step=global_step)

    pbar.close()
    wandb.run.summary["best_accuracy"] = max(acc_res) if acc_res else 0.0
    return acc_res, time.time() - start_time


def train_routine_pytorch(model, train_ds, val_ds, run_name, learning_rate, config, target_device="cuda:0", muon_learning_rate=0.02, optimizer_type="AdamW"):
    print(f"\nStarting run: {run_name} [{target_device}]")
    device = torch.device(target_device)
    model.to(device)

    batch_size = config.get("BATCH_SIZE", 32)
    grad_accum_steps = config.get("GRAD_ACCUM_STEPS", 1)
    train_steps = config.get("TRAIN_STEPS", 1000)
    log_freq = config.get("LOG_FREQ", 50)
    num_workers = 16 if torch.cuda.is_available() else 0

    collate_fn = DefaultDataCollator()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    if config.get("TRAIN_EPOCHS") is not None:
        train_steps = len(train_loader) * config["TRAIN_EPOCHS"] // grad_accum_steps

    if optimizer_type.lower() == "adamw":
        print(f"   Optimizer: AdamW (lr={learning_rate})")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=config.get("WEIGHT_DECAY", 0.0),
        )
    else:
        print(f"   Optimizer: Muon + AuxAdam (muon_lr={muon_learning_rate}, adam_lr={learning_rate})")
        muon_params = []
        adam_params = []
        matrices = [p for p in model.parameters() if p.requires_grad and p.ndim == 2 and p.size(0) > 1 and p.size(1) > 1]
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 2 and param.size(0) > 1 and param.size(1) > 1:
                continue
            adam_params.append(param)
        if matrices:
            adam_params.append(matrices[-1])
            muon_params.extend(matrices[:-1])
        optimizer = SingleDeviceMuonWithAuxAdam([
            dict(params=muon_params, use_muon=True, lr=muon_learning_rate, weight_decay=0.0),
            dict(params=adam_params, use_muon=False, lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.0),
        ])

    scaler = torch.amp.GradScaler("cuda")
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {trainable_params_count:,}")
    print(f"   Training for {train_steps} steps")

    acc_res, train_time = train_eval(
        train_loader,
        val_loader,
        train_steps,
        log_freq,
        run_name,
        model,
        optimizer,
        device,
        scaler,
        grad_accum_steps,
        config,
    )
    return {"name": run_name, "accuracy": acc_res, "train_time": train_time}


def run_multiset_pipeline(config_path="config.yaml", device="cuda:0", target_dataset=None, target_exp_index=None, override_model_id=None, target_seed=42):
    set_seed(target_seed)
    config = load_config(config_path)
    project_name = config.get("PROJECT_name", "gausslora")
    model_id = override_model_id or config.get("MODELS_TO_BENCHMARK", ["google/vit-base-patch16-224"])[0]
    optimizer_type = config.get("OPTIMIZER", "AdamW")
    datasets = config.get("DATASETS", ["stanfordcars"])
    subset_size = config.get("SUBSET_SIZE")
    experiments = config.get("EXPERIMENTS", [])
    save_dir = config.get("SAVE_DIR", "saved_adapters")
    group_id = f"run_{int(time.time())}"
    results = []

    print(f"Using model: {model_id}")

    for ds_name in datasets:
        if target_dataset is not None and ds_name != target_dataset:
            continue

        print(f"\n{'#' * 40}\nDataset: {ds_name.upper()}\n{'#' * 40}")
        train_ds, val_ds, num_labels = DatasetManager.get_dataset(ds_name, subset_size=subset_size, model_id=model_id)

        for exp_index, exp_cfg in enumerate(experiments):
            if target_exp_index is not None and exp_index != target_exp_index:
                continue

            torch.cuda.empty_cache()
            gc.collect()

            model_short = model_id.split("/")[-1]
            run_name = f"{model_short}_{ds_name}_{exp_cfg['name']}_seed{target_seed}"

            run = wandb.init(
                project=project_name,
                group=group_id,
                job_type=f"{model_short}_{ds_name}",
                name=run_name,
                config={"dataset": ds_name, "model": model_id, "seed": target_seed, **exp_cfg},
                reinit=True,
            )

            try:
                model = get_model(num_labels, exp_cfg, model_id=model_id)
                result = train_routine_pytorch(
                    model,
                    train_ds,
                    val_ds,
                    run_name=f"{ds_name}_{exp_cfg['name']}",
                    learning_rate=exp_cfg["lr"],
                    config=config,
                    target_device=device,
                    muon_learning_rate=exp_cfg.get("muon_learning_rate", 0.02),
                    optimizer_type=optimizer_type,
                )
                results.append(result)
                if result["accuracy"]:
                    print(f"Completed {exp_cfg['name']} -> final acc: {result['accuracy'][-1]:.4f}")

                if config.get("COLLECT_WEIGHTS", True):
                    save_adapter_weights(model, save_dir=save_dir, run_name=run_name.replace("/", "-"))

                del model
            except Exception as exc:
                print(f"Failed {exp_cfg['name']}: {exc}")
                import traceback
                traceback.print_exc()

            wandb.finish()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--exp_index", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_multiset_pipeline(
        config_path=args.config,
        device=args.device,
        target_dataset=args.dataset,
        target_exp_index=args.exp_index,
        override_model_id=args.model,
        target_seed=args.seed,
    )