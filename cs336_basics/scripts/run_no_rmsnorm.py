import torch

from cs336_basics.train import training_together


torch.set_float32_matmul_precision("high")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

BATCH_SIZE = 32
CONTEXT_LENGTH = 256

SHORT_TOTAL_STEPS = 500
SHORT_LRS = [3e-4, 1e-3, 2e-3, 3e-3]

RUN_SHORT_SWEEP = False
RUN_FULL_BEST = True
FULL_BEST_LR = 1e-3
FULL_TOTAL_STEPS = 5000


def run_one(alpha_max: float, total_steps: int, run_name: str):
    print(
        f"running no-RMSNorm {run_name}: "
        f"lr={alpha_max}, total_steps={total_steps}, device={DEVICE}, dtype={DTYPE}"
    )

    training_together(
        train_data_path="data/train.npy",
        val_data_path="data/val.npy",
        checkpoint_path=f"runs/no_rmsnorm/{run_name}/checkpoint.pt",

        vocab_size=10000,
        context_length=CONTEXT_LENGTH,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000.0,

        batch_size=BATCH_SIZE,
        total_steps=total_steps,

        alpha_max=alpha_max,
        alpha_min=alpha_max / 10,
        warmup_steps=50,
        cosine_steps=total_steps,

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
        save_final_checkpoint=False,
        use_rmsnorm=False,
    )


def main():
    if RUN_SHORT_SWEEP:
        for lr in SHORT_LRS:
            run_name = f"short_lr_{lr:.0e}_{SHORT_TOTAL_STEPS}steps"
            run_one(lr, SHORT_TOTAL_STEPS, run_name)

    if RUN_FULL_BEST:
        run_name = f"best_lr_{FULL_BEST_LR:.0e}_{FULL_TOTAL_STEPS}steps"
        run_one(FULL_BEST_LR, FULL_TOTAL_STEPS, run_name)


if __name__ == "__main__":
    main()
