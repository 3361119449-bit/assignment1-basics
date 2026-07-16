import torch

def save_checkpoint(model, optimizer, iteration, out):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )

# 即使 checkpoint 原来在 GPU 保存，也能先安全加载到 CPU。
def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, map_location="cpu")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]