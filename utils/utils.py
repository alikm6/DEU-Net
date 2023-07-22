import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def plot_img_and_mask(img: torch.Tensor, pred_mask: torch.Tensor, true_mask: None or torch.Tensor = None):
    img = img.cpu().detach().numpy()
    pred_mask = pred_mask.cpu().detach().numpy()
    if true_mask is not None:
        true_mask = true_mask.cpu().detach().numpy()

    assert true_mask is None or true_mask.shape == pred_mask.shape
    assert pred_mask.ndim == 2 or pred_mask.ndim == 3

    if pred_mask.ndim == 2:
        pred_mask = pred_mask[None, ...]

        if true_mask is not None:
            true_mask = true_mask[None, ...]

    classes = pred_mask.shape[0]

    total = classes + 1 if true_mask is None else 2 * classes + 1
    fig, ax = plt.subplots(1, total, constrained_layout=True, figsize=(6 * total, 6))

    for a in ax:
        a.axis('off')

    ax[0].set_title('Input image')
    ax[0].imshow(img.transpose((1, 2, 0)))

    if classes == 1:
        ax[1].set_title(f'Predicted mask')
        ax[1].imshow(pred_mask[0], cmap='gray')

        if true_mask is not None:
            ax[2].set_title(f'True mask')
            ax[2].imshow(true_mask[0], cmap='gray')
    else:
        for i in range(classes):
            ax[2 * i + 1].set_title(f'Predicted mask (class {i + 1})')
            ax[2 * i + 1].imshow(pred_mask[i], cmap='gray')

            if true_mask is not None:
                ax[2 * i + 2].set_title(f'True mask (class {i + 1})')
                ax[2 * i + 2].imshow(true_mask[i], cmap='gray')

    plt.show()


def mask_to_image(mask: torch.Tensor):
    mask = mask.cpu().detach().numpy()

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / (mask.shape[0] - 1)).astype(np.uint8))
