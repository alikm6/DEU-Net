from pathlib import Path
from PIL import Image

from tqdm import tqdm


def _load_resize_save(sub_dir_name, size, resampling_filter, input_ext, input_dir, output_dir):
    output_dir = output_dir / sub_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{sub_dir_name}:")

    for image_path in tqdm((input_dir / sub_dir_name).glob(f'*.{input_ext}')):
        img_pil = Image.open(image_path)
        img_pil = img_pil.resize(size, resampling_filter)
        img_pil.save(output_dir / (image_path.stem + ".png"), "PNG")

    print("")


def isic2016(dataset_dir, output_dir, size):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(dataset_dir)

    # train_images
    _load_resize_save(sub_dir_name='ISBI2016_ISIC_Part1_Training_Data',
                      size=size, resampling_filter=Image.BICUBIC, input_ext='jpg',
                      input_dir=dataset_dir, output_dir=output_dir)

    # train_masks
    _load_resize_save(sub_dir_name='ISBI2016_ISIC_Part1_Training_GroundTruth',
                      size=size, resampling_filter=Image.NEAREST, input_ext='png',
                      input_dir=dataset_dir, output_dir=output_dir)

    # test_images
    _load_resize_save(sub_dir_name='ISBI2016_ISIC_Part1_Test_Data',
                      size=size, resampling_filter=Image.BICUBIC, input_ext='jpg',
                      input_dir=dataset_dir, output_dir=output_dir)

    # test_masks
    _load_resize_save(sub_dir_name='ISBI2016_ISIC_Part1_Test_GroundTruth',
                      size=size, resampling_filter=Image.NEAREST, input_ext='png',
                      input_dir=dataset_dir, output_dir=output_dir)


def isic2017(dataset_dir, output_dir, size):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(dataset_dir)

    # train_images
    _load_resize_save(sub_dir_name='ISIC-2017_Training_Data',
                      size=size, resampling_filter=Image.BICUBIC, input_ext='jpg',
                      input_dir=dataset_dir, output_dir=output_dir)

    # train_masks
    _load_resize_save(sub_dir_name='ISIC-2017_Training_Part1_GroundTruth',
                      size=size, resampling_filter=Image.NEAREST, input_ext='png',
                      input_dir=dataset_dir, output_dir=output_dir)

    # val_images
    _load_resize_save(sub_dir_name='ISIC-2017_Validation_Data',
                      size=size, resampling_filter=Image.BICUBIC, input_ext='jpg',
                      input_dir=dataset_dir, output_dir=output_dir)

    # val_masks
    _load_resize_save(sub_dir_name='ISIC-2017_Validation_Part1_GroundTruth',
                      size=size, resampling_filter=Image.NEAREST, input_ext='png',
                      input_dir=dataset_dir, output_dir=output_dir)

    # test_images
    _load_resize_save(sub_dir_name='ISIC-2017_Test_v2_Data',
                      size=size, resampling_filter=Image.BICUBIC, input_ext='jpg',
                      input_dir=dataset_dir, output_dir=output_dir)

    # test_masks
    _load_resize_save(sub_dir_name='ISIC-2017_Test_v2_Part1_GroundTruth',
                      size=size, resampling_filter=Image.NEAREST, input_ext='png',
                      input_dir=dataset_dir, output_dir=output_dir)


def isic2018(dataset_dir, output_dir, size):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(dataset_dir)

    # train_images
    _load_resize_save(sub_dir_name='ISIC2018_Task1-2_Training_Input',
                      size=size, resampling_filter=Image.BICUBIC, input_ext='jpg',
                      input_dir=dataset_dir, output_dir=output_dir)

    # train_masks
    _load_resize_save(sub_dir_name='ISIC2018_Task1_Training_GroundTruth',
                      size=size, resampling_filter=Image.NEAREST, input_ext='png',
                      input_dir=dataset_dir, output_dir=output_dir)

    # val_images
    _load_resize_save(sub_dir_name='ISIC2018_Task1-2_Validation_Input',
                      size=size, resampling_filter=Image.BICUBIC, input_ext='jpg',
                      input_dir=dataset_dir, output_dir=output_dir)

    # val_masks
    _load_resize_save(sub_dir_name='ISIC2018_Task1_Validation_GroundTruth',
                      size=size, resampling_filter=Image.NEAREST, input_ext='png',
                      input_dir=dataset_dir, output_dir=output_dir)

    # test_images
    _load_resize_save(sub_dir_name='ISIC2018_Task1-2_Test_Input',
                      size=size, resampling_filter=Image.BICUBIC, input_ext='jpg',
                      input_dir=dataset_dir, output_dir=output_dir)

    # test_masks
    _load_resize_save(sub_dir_name='ISIC2018_Task1_Test_GroundTruth',
                      size=size, resampling_filter=Image.NEAREST, input_ext='png',
                      input_dir=dataset_dir, output_dir=output_dir)


if __name__ == '__main__':
    isic2016(dataset_dir="datasets/ISIC2016/", output_dir="datasets/ISIC2016_224x224/", size=(224, 224))

    isic2017(dataset_dir="datasets/ISIC2017/", output_dir="datasets/ISIC2017_224x224/", size=(224, 224))

    isic2018(dataset_dir="datasets/ISIC2018/", output_dir="datasets/ISIC2018_224x224/", size=(224, 224))
