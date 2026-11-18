# Training configuration

The training input is a reference pair and a coarse image at the target date. The supervised target is the observed fine image at that same target date. Use the same reflectance scaling and band order for all four images.

| Setting | Paper configuration |
| --- | --- |
| Patch size and batch size | 512 × 512, six patches per batch |
| Optimizer | AdamW, learning rate 0.0001, weight decay 0.0001 |
| Objective | Mean absolute error (L1) |
| Schedule | CosineAnnealingLR, up to 500 epochs, minimum learning rate 0.0000001 |
| Network defaults | 16 × 16 SDA windows, maximum drop-path rate 0.2 |
| Hardware | One NVIDIA A100 |

The following loop assumes that `train_loader` yields four floating-point tensors per batch, each in band-first format.

```python
import torch
from dsdf import DSDF

epochs = 500
model = DSDF(in_channels=6).cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

for epoch in range(epochs):
    for fine_ref, coarse_ref, coarse_target, fine_target in train_loader:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(fine_ref.cuda(), coarse_ref.cuda(), coarse_target.cuda())
        loss = torch.nn.functional.l1_loss(prediction, fine_target.cuda())
        loss.backward()
        optimizer.step()
    schedule.step()

torch.save(model.state_dict(), "dsdf_state_dict.pt")
```

This example saves the final epoch. To reproduce the paper's evaluation, apply the dataset splits, validation-based checkpoint selection and evaluation masks specified for the experiment. The reported metrics are computed on held-out imagery.
