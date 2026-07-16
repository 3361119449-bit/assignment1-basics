import math
import torch
torch.set_float32_matmul_precision("high")
# 第一，打开 TF32。适合 dtype=torch.float32 的 CUDA 训练：
from cs336_basics.train import training_together


# BATCH_SIZES = [1, 4, 16, 32, 64, 128, 256]
BATCH_SIZES = [32]
BASE_LR = 3e-3

# 如果你在 GPU 上跑，可以改大一点，比如 1000 或 2000
TOTAL_STEPS = 5000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
#     DTYPE = torch.bfloat16
# else:
#     DTYPE = torch.float32
DTYPE = torch.float32

def run_one(batch_size: int, lr: float):
    name = f"bs_{batch_size:03d}_lr_{lr:.0e}TOTAL_STEPS={TOTAL_STEPS}"

    print(f"running batch_size={batch_size}, lr={lr}, device={DEVICE}, dtype={DTYPE}")

    training_together(
        train_data_path="data/train.npy",
        val_data_path="data/val.npy",
        checkpoint_path=f"runs/batch_size_sweep/{name}/checkpoint.pt",

        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000.0,

        batch_size=batch_size,
        total_steps=TOTAL_STEPS,

        alpha_max=lr,
        alpha_min=lr / 10,
        warmup_steps=50,
        cosine_steps=TOTAL_STEPS,

        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        max_grad_norm=1.0,

        log_interval=10,
        eval_interval=100,
        checkpoint_interval=10**9,
        val_batches=10,

        device=DEVICE,
        dtype=DTYPE,
        resume_from=None,
        save_intermediate_checkpoints=False,
        save_final_checkpoint=True,
    )


def main():
    for batch_size in BATCH_SIZES:
        try:
            run_one(batch_size, BASE_LR)

        except RuntimeError as e:
            error = str(e).lower()

            if "out of memory" in error or "cuda" in error:
                print(f"OOM at batch_size={batch_size}. Stop or skip larger batch sizes.")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                break

            raise


if __name__ == "__main__":
    main()
