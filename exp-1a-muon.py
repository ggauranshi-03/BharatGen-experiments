import math
import argparse
import os
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, load_from_disk
# import pytorch_lightning as pl
# from lightning.pytorch.callbacks import Callback
# from lightning.pytorch.strategies import DDPStrategy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import DDPStrategy
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

from pytorch_lightning.loggers import WandbLogger
from transformers import get_linear_schedule_with_warmup
# ============================================================================ #
#                            Muon Optimizer Math                               #
# ============================================================================ #
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    if G.size(0) > G.size(1):
        G = G.t(); transposed = True
    else:
        transposed = False
    norm = G.norm() + eps
    X = G / norm
    X = X.bfloat16()
    for _ in range(steps):
        A = X.t() @ X
        B = b * A + c * A @ A
        X = X @ (a * torch.eye(X.size(1), device=X.device, dtype=X.dtype) + B)
    if transposed:
        X = X.t()
    return X.float()


def project_to_stiefel_tangent(theta, grad):
    theta = theta.to(grad.dtype)
    theta_t_grad = theta.t() @ grad
    symm_term = (theta_t_grad + theta_t_grad.t()) / 2.0
    return grad - (theta @ symm_term)

class Muon(Optimizer):
    def __init__(self, params, lr=0.0005, momentum=0.95, nesterov=True,
                 ns_steps=5, adam_w_lr=0.0002, adam_w_betas=(0.9, 0.999),
                 weight_decay=0.2, adam_w_weight_decay=0.0, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, adam_w_lr=adam_w_lr,
                        adam_w_betas=adam_w_betas, weight_decay=weight_decay, 
                        adam_w_weight_decay=adam_w_weight_decay, eps=eps)
        super().__init__(params, defaults)

    def _classify_param(self, p):
        is_embedding = (p.ndim == 2 and p.size(0) > 10000)
        return p.ndim == 2 and not is_embedding

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']; momentum = group['momentum']
            nesterov = group['nesterov']; ns_steps = group['ns_steps']
            weight_decay = group['weight_decay']; adam_lr = group['adam_w_lr']
            beta1, beta2 = group['adam_w_betas']; eps = group['eps']
            adam_w_wd = group['adam_w_weight_decay'] # Added distinct Adam WD

            for p in group['params']:
                grad = p.grad
                if grad is None: continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['use_muon'] = self._classify_param(p)
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                if state['use_muon']:
                    if weight_decay != 0: p.mul_(1 - lr * weight_decay)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                    g = buf.clone().add_(grad, alpha=1 - momentum) if nesterov else buf
                    g_ortho = zeropower_via_newtonschulz5(g, steps=ns_steps)
                    rows, cols = g.size()
                    g_ortho *= max(1, rows / cols) ** 0.5
                    p.add_(g_ortho, alpha=-lr)
                else:
                    if adam_w_wd != 0: p.mul_(1 - adam_lr * adam_w_wd) # Updated here
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bc1 = 1 - beta1 ** state['step']
                    bc2 = 1 - beta2 ** state['step']
                    step_size = adam_lr / bc1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss
# ============================================================================ #
#                           mu-GREATS Algorithms                               #
# ============================================================================ #

@torch.no_grad()
def greedy_omp(mu, K, kappa, steps=20):
    """
    Algorithm 2: GREEDYOMP (Vectorized & Accelerated)
    Selects `kappa` elements that maximize the mu-GREATS utility function.
    """
    batch_size = mu.size(0)
    
    # ---------------------------------------------------------
    # THE FIX: Cast K to float32 just for the norm calculation,
    # and use .item() to make lr_inner a native Python float.
    # ---------------------------------------------------------
    K_norm = torch.linalg.matrix_norm(K.to(torch.float32), ord=2).item()
    lr_inner = 1.0 / (K_norm + 1e-8)
    
    support = []
    weights = torch.zeros(batch_size, device=mu.device, dtype=mu.dtype)
    
    for _ in range(kappa):
        # Calculate marginal gains: gradient of utility w.r.t weights
        grad = mu - torch.matmul(K, weights)
        
        # Vectorized Masking
        if support:
            grad[support] = -float('inf')
            
        # Select best element
        best_idx = grad.argmax().item()
        support.append(best_idx)
        
        # Accelerated Projected Gradient Ascent (APGA)
        y = weights.clone()
        t = 1.0
        for _ in range(steps):
            g = mu - torch.matmul(K, y)
            
            weights_next = y.clone()
            # Vectorized update restricted to support
            weights_next[support] += lr_inner * g[support]
            weights_next = torch.clamp(weights_next, min=0.0)
            
            # Nesterov momentum update
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            y = weights_next + ((t - 1.0) / t_next) * (weights_next - weights)
            
            weights = weights_next
            t = t_next
            
    return support

class FastJohnsonLindenstrauss:
    """Random projection to compress (d x r) Stiefel tangent space to R^p"""
    def __init__(self, original_dim, p, device, dtype):
        self.p = p
        # Simple Gaussian random projection (approximate FJLT)
        self.proj_matrix = torch.randn(original_dim, p, device=device, dtype=dtype) / math.sqrt(p)
        
    def compress(self, x):
        # x is flattened grad [d * r]
        return torch.matmul(x, self.proj_matrix)

class LlamaLoRAMuGreatsModule(pl.LightningModule):
    def __init__(self, model_name, lora_config, lr, adam_w_lr, weight_decay, adam_w_weight_decay, momentum, adam_beta, kappa):
        super().__init__()
        self.save_hyperparameters(ignore=["lora_config"])
        self.automatic_optimization = False

        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        
        # --- CRITICAL MEMORY FIX 1: Gradient Checkpointing ---
        # This prevents the massive activation memory buildup
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads() 
        # -----------------------------------------------------

        self.model = get_peft_model(self.model, lora_config)
        self.fjlt = None 

    def _get_target_lora_param(self):
        for name, param in reversed(list(self.model.named_parameters())):
            if "v_proj.lora_A" in name and param.requires_grad:
                return param
        raise ValueError("Could not find a LoRA v_proj layer to track.")

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size = input_ids.size(0)
        
        kappa = min(self.hparams.kappa, max(1, batch_size - 1))
        target_param = self._get_target_lora_param()
        
        if self.fjlt is None:
            p_dim = self.model.config.hidden_size 
            self.fjlt = FastJohnsonLindenstrauss(target_param.numel(), p_dim, target_param.device, target_param.dtype)

        if batch_size < 4:
            loss = self(input_ids, attention_mask, labels=input_ids).loss
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt.step()
            return

        # --- CRITICAL MEMORY FIX 2: Isolate DDP ---
        # We explicitly use self.model instead of self to prevent Lightning's 
        # DDP wrapper from allocating massive network synchronization buckets
        base_model = self.model
        
        val_idx = 0
        val_input, val_mask = input_ids[val_idx:val_idx+1], attention_mask[val_idx:val_idx+1]
        train_input, train_mask = input_ids[1:], attention_mask[1:]
        candidate_size = train_input.size(0)
        
        # ------------------------------------------------------------------ #
        # STEP 1 & 2: Get Validation Gradients (Surgical extraction)         #
        # ------------------------------------------------------------------ #
        val_loss = base_model(input_ids=val_input, attention_mask=val_mask, labels=val_input).loss
        val_grad = torch.autograd.grad(val_loss, target_param)[0].detach().clone()
        
        with torch.no_grad():
            val_phi_compressed = self.fjlt.compress(project_to_stiefel_tangent(target_param, val_grad).flatten())
        del val_loss, val_grad # Instantly free the graph memory
        
        # ------------------------------------------------------------------ #
        # STEP 3: Get Per-Example Gradients for the Candidates               #
        # ------------------------------------------------------------------ #
        candidate_phis = []
        for i in range(candidate_size):
            ex_input, ex_mask = train_input[i:i+1], train_mask[i:i+1]
            ex_loss = base_model(input_ids=ex_input, attention_mask=ex_mask, labels=ex_input).loss
            
            # Surgical gradient calculation without modifying .grad attributes globally
            ex_grad = torch.autograd.grad(ex_loss, target_param)[0].detach().clone()
            
            with torch.no_grad():
                candidate_phis.append(self.fjlt.compress(project_to_stiefel_tangent(target_param, ex_grad).flatten()))
            del ex_loss, ex_grad # Instantly free the graph memory
            
        candidate_phis = torch.stack(candidate_phis) 
        
        # ------------------------------------------------------------------ #
        # STEP 4 & 5: Build mu, K, and Run GREEDYOMP                         #
        # ------------------------------------------------------------------ #
        K = torch.matmul(candidate_phis, candidate_phis.t())
        mu = torch.matmul(candidate_phis, val_phi_compressed)
        
        selected_relative_indices = greedy_omp(mu, K, kappa)
        selected_indices = [idx + 1 for idx in selected_relative_indices]
        
        # ------------------------------------------------------------------ #
        # STEP 6: Actual Forward/Backward on Selected Subset (DDP ENABLED)   #
        # ------------------------------------------------------------------ #
        # Now we use 'self' so DDP synchronizes the final gradients across all 4 GPUs
        final_input = input_ids[selected_indices]
        final_mask = attention_mask[selected_indices]
        
        final_loss = self(final_input, final_mask, labels=final_input).loss
        self.manual_backward(final_loss)
        self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt.step()
        
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        
        self.log("train/loss", final_loss, prog_bar=True)
        self.log("train/subset_selected", float(kappa))
        self.log("lr/muon", opt.param_groups[0]['lr'])
        if len(opt.param_groups) > 1:
            self.log("lr/adam", opt.param_groups[1]['lr'])

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = Muon(
            params,
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            adam_w_lr=self.hparams.adam_w_lr,
            adam_w_weight_decay=self.hparams.adam_w_weight_decay,
            adam_w_betas=(self.hparams.adam_beta, 0.999),
        )
        
        # Linear Scheduler Creation
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.05 * self.trainer.max_steps), # 5% warmup duration
            num_training_steps=self.trainer.max_steps,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
# ============================================================================ #
#                            Data & Main Setup                                 #
# ============================================================================ #
class TextDataset(Dataset):
    def __init__(self, tokenizer, data_dir, split="train", seq_length=2048):
        self.seq_length = seq_length
        
        # Load the Hugging Face DatasetDict from the directory
        dataset_dict = load_from_disk(data_dir)
        
        # Access the specific split ('train' or 'validation')
        raw = dataset_dict[split]

        # Standardize the target text column based on typical generated formats
        if "text" in raw.column_names:
            texts = [str(t) for t in raw["text"]] 
        elif "instruction" in raw.column_names and "output" in raw.column_names:
            texts = [f"Instruction: {row['instruction']}\nOutput: {row['output']}" for row in raw]
        else:
            texts = [str(row) for row in raw]

        tokens = tokenizer(
            texts,
            truncation=True,
            max_length=seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids = tokens["input_ids"]
        self.attention_mask = tokens["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        mask = self.attention_mask[idx]
        return {"input_ids": ids, "attention_mask": mask, "labels": ids.clone()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seq_length", type=int, default=1024)
    
    # Batch must be >= kappa for selection to make sense
    parser.add_argument("--micro_batch_size", type=int, default=8) 
    
    # -------------------------------------------------------------
    # Image Parameters applied as defaults here
    # -------------------------------------------------------------
    parser.add_argument("--kappa", type=int, default=4) 
    parser.add_argument("--lr", type=float, default=0.0005)              # Muon LR
    parser.add_argument("--weight_decay", type=float, default=0.2)       # Muon WD
    parser.add_argument("--momentum", type=float, default=0.95)          # Muon Momentum
    parser.add_argument("--adam_lr", type=float, default=0.0002)         # Adam LR
    parser.add_argument("--adam_weight_decay", type=float, default=0.01) # Adam WD
    parser.add_argument("--adam_beta", type=float, default=0.9)          # Adam Beta (momentum)
    parser.add_argument("--lora_dropout", type=float, default=0.05)      # LoRA Dropout
    
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--devices", type=int, default=2)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    # W&B Logger setup
    wandb_logger = WandbLogger(project="muon-greats-finetune", name="Exp-1a-Llama3-8B")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=128,
        lora_dropout=args.lora_dropout, # Inserted 0.05 from args
        target_modules=["q_proj", "k_proj", "v_proj",
                        "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    module = LlamaLoRAMuGreatsModule(
        model_name=args.model_name,
        lora_config=lora_config,
        lr=args.lr,
        adam_w_lr=args.adam_lr,
        weight_decay=args.weight_decay,
        adam_w_weight_decay=args.adam_weight_decay,
        momentum=args.momentum,
        adam_beta=args.adam_beta,
        kappa=args.kappa,
    )

    dataset = TextDataset(tokenizer, args.data_dir, split="train", seq_length=args.seq_length)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    strategy = DDPStrategy(find_unused_parameters=False) if args.devices > 1 else "auto"
    trainer = pl.Trainer(
        logger=wandb_logger,  # Pass wandb to trainer
        devices=args.devices,
        num_nodes=1,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        precision="bf16-true",
        log_every_n_steps=1,
    )

    print(f"\n{'='*70}")
    print(f"[START] Exp 1a: Llama3-8B LoRA + Muon + GREATS (W&B Active)")
    print(f"{'='*70}\n")

    trainer.fit(module, train_dataloaders=loader)

if __name__ == "__main__":
    main()
