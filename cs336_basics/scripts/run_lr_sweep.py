import torch

from cs336_basics.train import training_together


def run_one(alpha_max: float):
    alpha_min = alpha_max / 10
    name = f"lr_{alpha_max:.0e}"

    training_together(
        train_data_path="data/train.npy",
        val_data_path="data/val.npy",
        checkpoint_path=f"runs/{name}/checkpoint.pt",

        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000.0,

        batch_size=32,
        total_steps=500,

        alpha_max=alpha_max,
        alpha_min=alpha_min,
        warmup_steps=50,
        cosine_steps=500,

        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        max_grad_norm=1.0,

        log_interval=10,
        eval_interval=100,
        checkpoint_interval=500,
        val_batches=10,

        device="cpu",
        dtype=torch.float32,
        resume_from=None,
        save_intermediate_checkpoints=False,
        save_final_checkpoint=False,
    )


def main():
    # for lr in [1e-4, 3e-4, 1e-3, 3e-3]:
    # for lr in [1e-2, 3e-2, 1e-1]:
    for lr in [1.0]:
        print(f"running lr={lr}")
        run_one(lr)


if __name__ == "__main__":
    main()


#  cd /home/hapi/cs336/assignment1-basics
# uv run python -m cs336_basics.scripts.run_lr_sweep
