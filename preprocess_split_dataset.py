import random

from pathlib import Path


def isic2016(dataset_dir, output_dir, image_ext='jpg', mask_ext='png',
             val_ratio=0.125, split_random=True, random_seed=None):
    # The validation and training data are obtained by randomly splitting the original training data

    if random_seed is not None:
        random.seed(random_seed)

    if val_ratio > 1:
        raise ValueError("val_ratio is too big.")

    dataset_dir = Path(dataset_dir)

    train_val_images = list((dataset_dir / 'ISBI2016_ISIC_Part1_Training_Data').glob(f'*.{image_ext}'))
    train_val_masks = list((dataset_dir / 'ISBI2016_ISIC_Part1_Training_GroundTruth').glob(f'*.{mask_ext}'))

    if len(train_val_images) != len(train_val_masks) or len(train_val_images) == 0:
        raise ValueError("The set of images is empty or the number of masks is not equal to the number of images.")

    train_val_data = list(zip(train_val_images, train_val_masks))

    val_count = int(len(train_val_data) * val_ratio)
    train_count = len(train_val_data) - val_count

    if split_random:
        random.shuffle(train_val_data)

    train_data = train_val_data[:train_count]
    val_data = train_val_data[train_count:]

    test_images = list((dataset_dir / 'ISBI2016_ISIC_Part1_Test_Data').glob(f'*.{image_ext}'))
    test_masks = list((dataset_dir / 'ISBI2016_ISIC_Part1_Test_GroundTruth').glob(f'*.{mask_ext}'))

    if len(test_images) != len(test_masks) or len(test_images) == 0:
        raise ValueError("The set of images is empty or the number of masks is not equal to the number of images.")

    test_data = list(zip(test_images, test_masks))

    __save_output(output_dir, train_data, val_data, test_data)


def isic2017(dataset_dir, output_dir, image_ext='jpg', mask_ext='png'):
    dataset_dir = Path(dataset_dir)

    train_images = list((dataset_dir / 'ISIC-2017_Training_Data').glob(f'*.{image_ext}'))
    train_masks = list((dataset_dir / 'ISIC-2017_Training_Part1_GroundTruth').glob(f'*.{mask_ext}'))

    if len(train_images) != len(train_masks) or len(train_images) == 0:
        raise ValueError("The set of images is empty or the number of masks is not equal to the number of images.")

    train_data = list(zip(train_images, train_masks))

    val_images = list((dataset_dir / 'ISIC-2017_Validation_Data').glob(f'*.{image_ext}'))
    val_masks = list((dataset_dir / 'ISIC-2017_Validation_Part1_GroundTruth').glob(f'*.{mask_ext}'))

    if len(val_images) != len(val_masks) or len(val_images) == 0:
        raise ValueError("The set of images is empty or the number of masks is not equal to the number of images.")

    val_data = list(zip(val_images, val_masks))

    test_images = list((dataset_dir / 'ISIC-2017_Test_v2_Data').glob(f'*.{image_ext}'))
    test_masks = list((dataset_dir / 'ISIC-2017_Test_v2_Part1_GroundTruth').glob(f'*.{mask_ext}'))

    if len(test_images) != len(test_masks) or len(test_images) == 0:
        raise ValueError("The set of images is empty or the number of masks is not equal to the number of images.")

    test_data = list(zip(test_images, test_masks))

    __save_output(output_dir, train_data, val_data, test_data)


def isic2018(dataset_dir, output_dir, image_ext='jpg', mask_ext='png',
             train_ratio=0.7, val_ratio=0.1, split_random=True, random_seed=None):
    # we only use the training data of this dataset!

    if random_seed is not None:
        random.seed(random_seed)

    if train_ratio + val_ratio > 1:
        raise ValueError("train_ratio + val_ratio is too big.")

    dataset_dir = Path(dataset_dir)

    train_val_test_images = list((dataset_dir / 'ISIC2018_Task1-2_Training_Input').glob(f'*.{image_ext}'))
    train_val_test_masks = list((dataset_dir / 'ISIC2018_Task1_Training_GroundTruth').glob(f'*.{mask_ext}'))

    if len(train_val_test_images) != len(train_val_test_masks) or len(train_val_test_images) == 0:
        raise ValueError("The set of images is empty or the number of masks is not equal to the number of images.")

    train_val_test_data = list(zip(train_val_test_images, train_val_test_masks))

    train_count = int(len(train_val_test_data) * train_ratio)
    val_count = int(len(train_val_test_data) * val_ratio)
    # test_count = len(train_val_test_data) - train_count - val_count

    if split_random:
        random.shuffle(train_val_test_data)

    train_data = train_val_test_data[:train_count]
    val_data = train_val_test_data[train_count:train_count + val_count]
    test_data = train_val_test_data[train_count + val_count:]

    __save_output(output_dir, train_data, val_data, test_data)


def ph2(dataset_dir, output_dir, image_ext='bmp', mask_ext='bmp',
        train_ratio=0.7, val_ratio=0.1, split_random=True, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    if train_ratio + val_ratio > 1:
        raise ValueError("train_ratio + val_ratio is too big.")

    dataset_dir = Path(dataset_dir)

    train_val_test_images = []
    train_val_test_masks = []

    for tmp_path in (dataset_dir / 'PH2 Dataset images').iterdir():
        train_val_test_images.append(
            tmp_path / ("%s_Dermoscopic_Image" % tmp_path.name) / f"{tmp_path.name}.{image_ext}")
        train_val_test_masks.append(tmp_path / ("%s_lesion" % tmp_path.name) / f"{tmp_path.name}_lesion.{mask_ext}")

    train_val_test_data = list(zip(train_val_test_images, train_val_test_masks))

    train_count = int(len(train_val_test_data) * train_ratio)
    val_count = int(len(train_val_test_data) * val_ratio)
    # test_count = len(train_val_test_data) - train_count - val_count

    if split_random:
        random.shuffle(train_val_test_data)

    train_data = train_val_test_data[:train_count]
    val_data = train_val_test_data[train_count:train_count + val_count]
    test_data = train_val_test_data[train_count + val_count:]

    __save_output(output_dir, train_data, val_data, test_data)


def __save_output(output_dir, train_data, val_data, test_data):
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if len(train_data) != 0:
        fp_images = open(output_dir.joinpath("train_images.txt"), 'w')
        fp_masks = open(output_dir.joinpath("train_masks.txt"), 'w')

        for item in train_data:
            fp_images.write("%s\n" % item[0])
            fp_masks.write("%s\n" % item[1])

        fp_images.close()
        fp_masks.close()

    if len(val_data) != 0:
        fp_images = open(output_dir.joinpath("val_images.txt"), 'w')
        fp_masks = open(output_dir.joinpath("val_masks.txt"), 'w')

        for item in val_data:
            fp_images.write("%s\n" % item[0])
            fp_masks.write("%s\n" % item[1])

        fp_images.close()
        fp_masks.close()

    if len(test_data) != 0:
        fp_images = open(output_dir.joinpath("test_images.txt"), 'w')
        fp_masks = open(output_dir.joinpath("test_masks.txt"), 'w')

        for item in test_data:
            fp_images.write("%s\n" % item[0])
            fp_masks.write("%s\n" % item[1])

        fp_images.close()
        fp_masks.close()


if __name__ == '__main__':
    random_seed = 1234

    isic2016(dataset_dir="datasets/ISIC2016/", output_dir="data/isic2016/",
             random_seed=random_seed)
    isic2016(dataset_dir="datasets/ISIC2016_224x224/", output_dir="data/isic2016_224x224/", image_ext='png',
             random_seed=random_seed)

    isic2017(dataset_dir="datasets/ISIC2017/", output_dir="data/isic2017/")
    isic2017(dataset_dir="datasets/ISIC2017_224x224/", output_dir="data/isic2017_224x224/", image_ext='png')

    isic2018(dataset_dir="datasets/ISIC2018/", output_dir="data/isic2018/",
             random_seed=random_seed)
    isic2018(dataset_dir="datasets/ISIC2018_224x224/", output_dir="data/isic2018_224x224/", image_ext='png',
             random_seed=random_seed)

    ph2(dataset_dir="datasets/PH2/", output_dir="data/ph2/",
        random_seed=random_seed)
