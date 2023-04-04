import albumentations
from albumentations.pytorch import ToTensorV2


def get_train_transforms(params):
    train_transforms = albumentations.Compose(
        [
            albumentations.Resize(height=params.image_size, width=params.image_size),
            albumentations.HorizontalFlip(),
            albumentations.OneOf(
                [
                    albumentations.RandomContrast(),
                    albumentations.RandomBrightness(),
                ]
            ),
            albumentations.ShiftScaleRotate(rotate_limit=20, scale_limit=0.2),
            albumentations.JpegCompression(80),
            albumentations.HueSaturationValue(),
            albumentations.Normalize(),
            ToTensorV2(),
        ]
    )
    return train_transforms


def get_test_transforms(params):
    test_transforms = albumentations.Compose(
        [
            albumentations.Resize(height=params.image_size, width=params.image_size),
            albumentations.Normalize(),
            ToTensorV2(),
        ]
    )
    return test_transforms
