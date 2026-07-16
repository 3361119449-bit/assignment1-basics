import torch
from cs336_basics.train import training_together



def main():
    print("开始运行训练")
    model, optimizer = training_together(
        train_data_path="data/train.npy",
        val_data_path="data/val.npy",
        checkpoint_path="runs/baseline_001/checkpoint.pt",

        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000.0,

        batch_size=32,
        total_steps=5000,

        alpha_max=3e-4,
        alpha_min=3e-5,
        warmup_steps=500,
        cosine_steps=5000,

        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        max_grad_norm=1.0,

        log_interval=10,
        eval_interval=100,
        checkpoint_interval=1000,
        val_batches=20,

        device="cuda",
        dtype=torch.bfloat16,
        resume_from=None,
    )



if __name__ == "__main__":
    main()





# 最好的 learning rate 通常接近模型刚好不发散的边界。
# 现在让我们改变 batch size，



# 低资源正式跑：

# device="cpu"
# dtype=torch.float32
# d_model=512
# num_layers=4
# num_heads=16
# d_ff=1344
# batch_size=32
# total_steps=5000
# warmup_steps=500
# cosine_steps=5000
# GPU/B200 正式作业 baseline：

# device="cuda"
# dtype=torch.bfloat16
# d_model=512
# num_layers=4
# num_heads=16
# d_ff=1344
# batch_size=32
# total_steps=40000
# warmup_steps=4000
# cosine_steps=40000