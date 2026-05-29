"""Image preprocessing.

No augmentations are applied: the dataset is already augmented offline by its author.
"""

from torchvision import transforms

# ImageNet statistics — used because pretrained backbones (future work) expect them,
# and they also work fine for a from-scratch CNN as a sensible normalization scheme.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_preprocessing(img_size: int) -> transforms.Compose:
    """Resize + ToTensor + ImageNet normalization. Same for train / val / test."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
