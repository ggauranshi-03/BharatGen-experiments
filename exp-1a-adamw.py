import math
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, load_from_disk
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import DDPStrategy
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from pytorch_lightning.loggers import WandbLogger
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

# ============================================================================ #
#                            GREATS Algorithms                                 #
# ============================================================================ #

@torch.no_grad()
def greedy_omp(mu, K, kappa, steps=20):
    """
    Algorithm 2: GREEDYOMP (Vectorized & Accelerated)
    Selects `kappa` elements that maximize the GREATS utility function.
    """
    batch_size = mu.size(0)
    
    # ---------------------------------------------------------
    # Cast K to float32 just for the norm calculation,
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
    """Random projection to compress the gradient space to R^p"""
    def __init__(self, original_dim, p, device, dtype):
        self.p = p
        # Simple Gaussian random projection (approximate FJLT)
        self.proj_matrix = torch.randn(original_dim, p, device=device, dtype=dtype) / math.sqrt(p)
        
    def compress(self, x):
        # x is flattened grad
        return torch.matmul(x, self.proj_matrix)


# ============================================================================ #
#                            Lightning Module                                  #
# ============================================================================ #

class LlamaLoRAGreatsModule(pl.LightningModule):
    def __init__(self, model_name, lora_config, lr, weight_decay, beta1, beta2, kappa):
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
            # Standard Euclidean flattened gradient
            val_phi_compressed = self.fjlt.compress(val_grad.flatten())
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
                candidate_phis.append(self.fjlt.compress(ex_grad.flatten()))
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
        # Now we use 'self' so DDP synchronizes the final gradients across GPUs
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
        self.log("lr/adamw", opt.param_groups[0]['lr'])

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(self.hparams.beta1, self.hparams.beta2),
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
    # Hyperparameters
    # -------------------------------------------------------------
    parser.add_argument("--kappa", type=int, default=4) 
    parser.add_argument("--lr", type=float, default=0.0002)              # AdamW LR
    parser.add_argument("--weight_decay", type=float, default=0.01)      # AdamW WD
    parser.add_argument("--beta1", type=float, default=0.9)              # AdamW Beta1
    parser.add_argument("--beta2", type=float, default=0.999)            # AdamW Beta2
    parser.add_argument("--lora_dropout", type=float, default=0.05)      # LoRA Dropout
    
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--devices", type=int, default=2)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    # W&B Logger setup
    wandb_logger = WandbLogger(project="exp-1a", name="Exp-1a_adamw")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=128,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj",
                        "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    module = LlamaLoRAGreatsModule(
        model_name=args.model_name,
        lora_config=lora_config,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
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
        logger=wandb_logger,
        devices=args.devices,
        num_nodes=1,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        precision="bf16-true",
        log_every_n_steps=1,
    )

    print(f"\n{'='*70}")
    print(f"[START] Exp 1b: Llama3-8B LoRA + AdamW + GREATS (W&B Active)")
    print(f"{'='*70}\n")

    trainer.fit(module, train_dataloaders=loader)

if __name__ == "__main__":
    main()
