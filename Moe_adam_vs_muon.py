import os
import math
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import (
    AutoTokenizer,
    MixtralConfig,
    MixtralForCausalLM,
    get_cosine_schedule_with_warmup
)
from datasets import load_from_disk
from subset_select import create_subset_selector


# ============================================================================ #
#                            Dataset Preparation                               #
# ============================================================================ #
class TextDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=1024):
        self.max_length = max_length
        safe_dir_name = os.path.basename(os.path.normpath(data_dir))
        cache_file = f"local_cache_{safe_dir_name}.bin"

        if os.path.exists(cache_file):
            print(f"Loading cached tokens from {cache_file}...")
            self.tokens = torch.load(cache_file)
        else:
            print(f"Loading local dataset from {data_dir}...")
            raw_dataset = load_from_disk(data_dir)
            data_to_process = raw_dataset["train"] if "train" in raw_dataset else raw_dataset

            if "text" in data_to_process.column_names:
                texts = data_to_process["text"]
            elif "instruction" in data_to_process.column_names and "output" in data_to_process.column_names:
                texts = [f"Instruction: {row['instruction']}\nOutput: {row['output']}" for row in data_to_process]
            else:
                texts = [str(row) for row in data_to_process]

            self.tokens = []
            for text in tqdm(texts, desc="Tokenizing"):
                if text and str(text).strip():
                    self.tokens.extend(tokenizer.encode(str(text), add_special_tokens=True))
            torch.save(self.tokens, cache_file)

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        s = idx * self.max_length
        return torch.tensor(self.tokens[s: s + self.max_length], dtype=torch.long)


# ============================================================================ #
#                           Muon Optimizer                                     #
# ============================================================================ #
@torch.compile
def _zeropower_ns(G, steps):
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, muon_params, lr=0.0005, momentum=0.95, nesterov=True,
                 ns_steps=5, adamw_params=None, adamw_lr=2e-4,
                 adamw_betas=(0.9, 0.99), adamw_eps=1e-8, adamw_wd=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, adamw_betas=adamw_betas,
                        adamw_eps=adamw_eps, adamw_wd=adamw_wd)
        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        for p in muon_params:
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    @staticmethod
    def _lr_scale(lr, shape):
        A, B = shape[:2]
        return lr * 0.2 * math.sqrt(max(A, B))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]

            for p in [p for p in group["params"] if self.state[p].get("use_muon")]:
                g = p.grad
                if g is None: continue
                if g.ndim > 2: g = g.view(g.size(0), -1)
                st = self.state[p]
                if "buf" not in st: st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                u = _zeropower_ns(g, steps=group["ns_steps"])
                # FIX: decouple weight decay from LR scaling — use fixed small WD
                p.data.mul_(1 - 0.01 * lr)
                p.data.add_(u, alpha=-self._lr_scale(lr, p.shape))

            beta1, beta2 = group["adamw_betas"]
            eps, adamw_wd = group["adamw_eps"], group["adamw_wd"]
            for p in [p for p in group["params"] if not self.state[p].get("use_muon")]:
                g = p.grad
                if g is None: continue
                st = self.state[p]
                if "step" not in st:
                    st.update(dict(step=0, m1=torch.zeros_like(g), m2=torch.zeros_like(g)))
                st["step"] += 1
                st["m1"].lerp_(g, 1 - beta1)
                st["m2"].lerp_(g.square(), 1 - beta2)
                g_hat = st["m1"] / (eps + st["m2"].sqrt())
                bc = (1 - beta1 ** st["step"]) / (1 - beta2 ** st["step"]) ** 0.5
                p.data.mul_(1 - lr * adamw_wd)
                p.data.add_(g_hat, alpha=-lr / bc)
        return loss


# ============================================================================ #
#                           Checkpoint Helpers                                 #
# ============================================================================ #
def save_checkpoint(epoch, global_step, model, optimizer, scheduler, args, local_rank):
    """Save checkpoint from rank 0 only. Uses dist.barrier() so all ranks sync."""
    if local_rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(
            args.checkpoint_dir,
            f"ckpt_epoch{epoch:04d}_step{global_step}.pt"
        )
        torch.save({
            "epoch":                epoch,
            "global_step":          global_step,
            "model_state_dict":     model.module.state_dict(),  # unwrap DDP
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args":                 vars(args),
        }, ckpt_path)
        print(f"\n[Epoch {epoch}] Checkpoint saved → {ckpt_path}")
        if wandb.run:
            wandb.save(ckpt_path)
    dist.barrier()  # all ranks wait until rank 0 finishes writing


def load_checkpoint(args, model, optimizer, scheduler, local_rank):
    """Load checkpoint and return (start_epoch, global_step)."""
    if args.resume_from is None or not os.path.isfile(args.resume_from):
        return 0, 0
    map_location = {"cuda:0": f"cuda:{local_rank}"}
    ckpt = torch.load(args.resume_from, map_location=map_location, weights_only=False)
    model.module.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch  = ckpt["epoch"]
    global_step  = ckpt["global_step"]
    if local_rank == 0:
        print(f"Resumed from {args.resume_from} (epoch {start_epoch}, step {global_step})")
    return start_epoch, global_step


# ============================================================================ #
#                                 Main Loop                                    #
# ============================================================================ #
def main():
    # ── DDP init ─────────────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size  = dist.get_world_size()
    device      = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if local_rank != 0:
        os.environ["WANDB_MODE"] = "disabled"

    # ── Args ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer",        type=str,   default="muon",      choices=["muon", "adamw"])
    parser.add_argument("--mode",             type=str,   default="mu-greats")
    parser.add_argument("--batch_size",       type=int,   default=8)
    parser.add_argument("--lr",               type=float, default=5e-4)       # FIX: reduced from 1e-3
    parser.add_argument("--adam_lr",          type=float, default=2e-4)
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--wandb_project",    type=str,   default="nemo-moe-muon")
    parser.add_argument("--data_dir",         type=str,   default="data")
    parser.add_argument("--checkpoint_dir",   type=str,   default="checkpoints")
    parser.add_argument("--checkpoint_every", type=int,   default=25)
    parser.add_argument("--resume_from",      type=str,   default=None)
    parser.add_argument("--num_val_samples",  type=int,   default=8)
    args = parser.parse_args()

    if local_rank == 0:
        wandb.init(project=args.wandb_project, name=f"MoE-1B_{args.optimizer}_{args.mode}")

    # ── Model ─────────────────────────────────────────────────────────────────
    config = MixtralConfig(
        vocab_size=50257,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=10,
        num_attention_heads=16,
        num_key_value_heads=4,
        num_local_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=1024,
        router_aux_loss_coef=0.01,
        output_router_logits=True
    )
    model = MixtralForCausalLM(config)
    model.gradient_checkpointing_enable()
    model = model.to(torch.bfloat16).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if local_rank == 0:
        print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # ── Dataset ───────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    full_ds   = TextDataset(args.data_dir, tokenizer, max_length=1024)

    eval_size  = max(1, int(len(full_ds) * 0.1))
    train_size = len(full_ds) - eval_size
    train_ds   = Subset(full_ds, range(train_size))
    eval_ds    = Subset(full_ds, range(train_size, len(full_ds)))

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    eval_sampler  = DistributedSampler(eval_ds,  num_replicas=world_size, rank=local_rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              pin_memory=True, num_workers=4)
    eval_loader  = DataLoader(eval_ds,  batch_size=args.batch_size, sampler=eval_sampler,
                              pin_memory=True, num_workers=4)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if args.optimizer == "muon":
        muon_p  = [p for n, p in model.named_parameters()
                   if p.ndim >= 2 and "embed_tokens" not in n and "lm_head" not in n]
        adamw_p = [p for n, p in model.named_parameters()
                   if not (p.ndim >= 2 and "embed_tokens" not in n and "lm_head" not in n)]
        optimizer = Muon(muon_p, lr=args.lr, adamw_params=adamw_p, adamw_lr=args.adam_lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.adam_lr, weight_decay=0.1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * args.epochs
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch, global_step = load_checkpoint(args, model, optimizer, scheduler, local_rank)

    # ── Subset Selector ───────────────────────────────────────────────────────
    base_model = model.module
    selector = create_subset_selector(
        mode=args.mode,
        model=base_model,
        val_dataloader=eval_loader,
        device=device,
        batch_size=args.batch_size,
        num_val_samples=args.num_val_samples,
        current_lr=optimizer.param_groups[0]["lr"]
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(local_rank != 0)):
            batch = batch.to(device)

            # Subset selection (ghost backward inside no_sync to avoid DDP conflicts)
            if selector is not None:
                if hasattr(selector, "current_lr"):
                    selector.current_lr = optimizer.param_groups[0]["lr"]
                batch_dict = {"input_ids": batch}
                with model.no_sync():
                    sel_indices = selector.select(batch_dict, base_model=base_model)
                batch = batch[sel_indices]

            # Forward + backward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=batch, labels=batch)
                loss    = outputs.loss

            loss.backward()

            grad_norm = math.sqrt(sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ))
            train_ppl = math.exp(min(loss.item(), 100))

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if local_rank == 0:
                wandb.log({
                    "train/loss":       loss.item(),
                    "train/perplexity": train_ppl,
                    "train/grad_norm":  grad_norm,
                    "learning_rate":    optimizer.param_groups[0]["lr"],
                    "epoch":            epoch + 1,
                    "global_step":      global_step,
                }, step=global_step)

            # Periodic eval
            if global_step % 100 == 0:
                model.eval()
                eval_loss = 0.0
                with torch.no_grad():
                    for ev_batch in eval_loader:
                        ev_batch = ev_batch.to(device)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            ev_outputs = model(input_ids=ev_batch, labels=ev_batch)
                        eval_loss += ev_outputs.loss.item()
                        break  # fast single-batch eval
                if local_rank == 0:
                    wandb.log({
                        "eval/loss":       eval_loss,
                        "eval/perplexity": math.exp(min(eval_loss, 100))
                    }, step=global_step)
                model.train()

        # ── Checkpoint every N epochs ────────────────────────────────────────
        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(epoch + 1, global_step, model, optimizer, scheduler, args, local_rank)

    # ── Save final checkpoint ─────────────────────────────────────────────────
    save_checkpoint(args.epochs, global_step, model, optimizer, scheduler, args, local_rank)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
