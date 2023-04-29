import albumentations as A
from albumentations.pytorch import ToTensorV2


class Augmentations:

    def no_aug():
        """No augmentations
        Returns: Compose
        """
        return A.Compose(
            [
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    
    def vertical_flip():
        """Vertical flip
        Returns: Compose
        """
        return A.Compose(
            [
                A.VerticalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def horizontal_flip():
        """Horizontal flip
        Returns: Compose
        """
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    
    def image_aug():
        """Focus on image augmentations like sharpen, brightness and contrast,
            Returns: Compose"""
        return A.Compose(
            [   A.OneOf(
                    [
                        #A.Blur(blur_limit=5, p=0.5),
                        A.Sharpen(p=0.5)
                    ]
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    
    def geometric_aug():
        """Focus on geometric augmentations like rotation, scale, crop, and flip
            Returns: Compose"""
        return A.Compose(
            [
                #A.RandomScale(scale_limit=(-0.5, 1), p=1),
                A.RandomCrop(200,200, p=0.5),
                A.PadIfNeeded(256,256),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def weak_aug():
        """The standard st++ augmentations
        Returns: Compose
        """
        return A.Compose(
            [
                A.RandomScale(scale_limit=(-0.5, 1), p=1),
                A.PadIfNeeded(256,256),
                #A.RandomCrop(64,64),           
                A.VerticalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    


    def strong_aug():
        """A strong augmentations pipeline based on https://github.com/sneddy/pneumothorax-segmentation
        Returns: Compose
        """
        return A.Compose(
            [
                A.PadIfNeeded(256,256),
                #A.SmallestMaxSize(64),
                #A.CenterCrop(32,32),
                A.VerticalFlip(p=0.5),
                A.OneOf(
                    [
                        A.RandomContrast(p=0.5, limit=[-0.2, 0.2]),
                        A.RandomGamma(p=0.5, gamma_limit=[80, 120]),
                        A.RandomBrightness(p=0.5, limit=[-0.2, 0.2]),
                    ]
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            p=0.5,
                            alpha=120,
                            sigma=6.0,
                            alpha_affine=3.5999999999999996,
                            interpolation=1,
                            border_mode=4,
                            value=None,
                            mask_value=None,
                            approximate=False,
                        ),
                        A.GridDistortion(
                            p=0.5,
                            num_steps=5,
                            distort_limit=[-0.3, 0.3],
                            interpolation=1,
                            border_mode=4,
                            value=None,
                            mask_value=None,
                        ),
                        A.OpticalDistortion(
                            p=0.5,
                            distort_limit=[-2, 2],
                            shift_limit=[-0.5, 0.5],
                            interpolation=1,
                            border_mode=4,
                            value=None,
                            mask_value=None,
                        ),
                    ]
                ),
                A.ShiftScaleRotate(
                    p=0.5,
                    shift_limit=[-0.0625, 0.0625],
                    scale_limit=[-0.09999999999999998, 0.10000000000000009],
                    rotate_limit=[-45, 45],
                    interpolation=1,
                    border_mode=4,
                    value=None,
                    mask_value=None,
                ),
                A.Normalize(),
                ToTensorV2(),
            ]
        )