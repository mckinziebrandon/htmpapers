import math
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def random_color_jitter(
    brightness: float = 0,
    contrast: float = 0,
    hue: float = 0,
    saturation: float = 0,
):
    return transforms.ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
    )


def random_affine(
    max_rotation: float = 0.2,
    horizontal_shear: float = 0.2,
    vertical_shear: float = 0.2
):
    return transforms.RandomAffine(
        degrees=math.degrees(max_rotation),
        shear=(0, math.degrees(horizontal_shear), 0, math.degrees(vertical_shear)),
        interpolation=InterpolationMode.BILINEAR)


def get_dataset_transform(split: str, mean, std):
    transforms_list = []
    if split == 'train':
        transforms_list.extend([
            transforms.RandomHorizontalFlip(),
            random_affine(),
            random_color_jitter(0.2, 0.1, 0.025)])

    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transforms.Compose(transforms_list)
