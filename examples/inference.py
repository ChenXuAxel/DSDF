"""Run one synthetic six-band fusion example on CUDA."""
import torch

from dsdf import DSDF


def main():
    torch.manual_seed(42)
    model = DSDF(in_channels=6).cuda().eval()
    fine_reference, coarse_reference, coarse_target = (
        torch.rand(1, 6, 64, 64, device="cuda") for _ in range(3)
    )
    with torch.no_grad():
        prediction = model(fine_reference, coarse_reference, coarse_target)
    assert prediction.shape == fine_reference.shape
    assert torch.isfinite(prediction).all()
    print(f"Predicted fine image: {tuple(prediction.shape)}")


if __name__ == "__main__":
    main()
