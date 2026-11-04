import os
import torch
import logging
import numpy as np
from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as transforms
from torch.distributions import (
    Independent,
    ReshapeTransform,
    TransformedDistribution,
    Categorical,
    Normal,
    MixtureSameFamily,
)


class ImageNet64(CIFAR10):
    """ImageNet64 dataset stored in the CIFAR-style batch format."""

    base_folder = "imagenet64"
    url = None
    filename = None
    tgz_md5 = None
    train_list = [
        ["train_data_batch_1", ""],
        ["train_data_batch_2", ""],
        ["train_data_batch_3", ""],
        ["train_data_batch_4", ""],
        ["train_data_batch_5", ""],
        ["train_data_batch_6", ""],
        ["train_data_batch_7", ""],
        ["train_data_batch_8", ""],
        ["train_data_batch_9", ""],
        ["train_data_batch_10", ""],
    ]
    test_list = [["val_data", ""]]
    meta = {}

    def _load_meta(self):
        # CIFAR10 assumes 32x32 images. The ImageNet64 batches are flattened 64x64
        # images, so we undo the intermediate 32x32 reshape before returning HWC.
        self.data = self.data.transpose((0, 3, 1, 2))
        self.data = self.data.reshape(-1, 3, 64, 64)
        self.data = self.data.transpose((0, 2, 3, 1))

    def _check_integrity(self) -> bool:
        # The upstream md5 check is not available for these local batch files.
        return True


def _build_imagenet64_transform(config):

    transform_steps = []
    transform_steps.append(transforms.RandomHorizontalFlip())
    transform_steps.append(transforms.ToTensor())
    return transforms.Compose(transform_steps)


def _build_eval_imagenet64_transform(config):
    transform_steps = []
    transform_steps.append(transforms.ToTensor())
    return transforms.Compose(transform_steps)


def _build_always_flip_eval_imagenet64_transform(config):
    transform_steps = []
    transform_steps.append(transforms.RandomHorizontalFlip(p=1.0))
    transform_steps.append(transforms.ToTensor())
    return transforms.Compose(transform_steps)


def create_mnist_mixture_distribution(device):
    dataset = MNIST(
        "../data", train=True, download=True, transform=transforms.ToTensor()
    )

    indices = torch.load("data_cache/mnist_mixture_indices.pt")
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=100, shuffle=False)

    images, _ = next(iter(dataloader))

    images = images.reshape((100, 784)).to(device=device)

    mix = Categorical(probs=torch.full((100,), 1 / 100, device=device))

    comps = Independent(
        Normal(images, torch.full_like(images, 0.1)), reinterpreted_batch_ndims=1
    )
    base_dist = MixtureSameFamily(mix, comps)

    transform = ReshapeTransform((784,), (1, 28, 28))
    mnist_dist = TransformedDistribution(base_dist, [transform])

    data_statistics = {"means": images, "std": 0.1}

    return mnist_dist, data_statistics


def prepare_data_cache(config):
    if config.data.name == "mnist_mixture":
        dataset = MNIST(
            "../data", train=True, download=True, transform=transforms.ToTensor()
        )

        # 2. Select indices for 10 samples per class
        def get_balanced_subset_indices(dataset, samples_per_class=10):
            targets = np.array(dataset.targets)
            indices = []

            for class_idx in range(10):  # MNIST has 10 classes (0-9)
                # Find where the labels equal the current class
                class_indices = np.where(targets == class_idx)[0]
                # Take the first N samples (or random choice if preferred)
                selected = class_indices[:samples_per_class]
                indices.extend(selected)

            return indices

        indices = get_balanced_subset_indices(dataset, samples_per_class=10)
        indices = [int(i) for i in indices]
        if not os.path.exists("data_cache"):
            os.mkdir("data_cache")
        save_path = "data_cache/mnist_mixture_indices.pt"
        if os.path.exists(save_path):
            logging.info(f"Removing existing data cache at {save_path}")
            os.remove(save_path)

        torch.save(indices, save_path)

    return


def get_is_set_path(config):

    if config.data.name == "imagenet64":
        return f"is_sets/imagenet64_standard_normal_{config.seed}"

    raise ValueError(
        f"Importance-sampling dataset path is not defined for {config.data.name}."
    )


def prepare_data_sets(config):
    if config.data.name == "mnist_mixture":
        mnist_dist, _ = create_mnist_mixture_distribution(device=config.device)

        is_infinite = config.data.is_infinite
        is_infinite_string = "infinite" if is_infinite else "finite"

        batch_size = config.eval.batch_size
        data_shape = tuple(config.data.shape)

        def sample_tensor_dataset(num_samples):
            samples = torch.empty((num_samples, *data_shape), dtype=torch.float32)
            offset = 0
            while offset < num_samples:
                current_batch_size = min(batch_size, num_samples - offset)
                samples[offset : offset + current_batch_size] = mnist_dist.sample(
                    (current_batch_size,)
                ).cpu()
                offset += current_batch_size

            return samples

        if not is_infinite:
            num_samples = 50000  # number of training samples to generate

            if not os.path.exists("train_sets"):
                os.mkdir("train_sets")
            train_set_path = (
                f"train_sets/mnist_mixture_{is_infinite_string}_{config.seed}.pt"
            )
            if os.path.exists(train_set_path):
                logging.info(
                    f"Removing existing training set cache at {train_set_path}"
                )
                os.remove(train_set_path)

            train_set = sample_tensor_dataset(num_samples)
            torch.save(train_set, train_set_path)
            del train_set

        if not os.path.exists("val_sets"):
            os.mkdir("val_sets")
        if not os.path.exists("test_sets"):
            os.mkdir("test_sets")
        if not os.path.exists("is_sets"):
            os.mkdir("is_sets")
        val_set_path = f"val_sets/mnist_mixture_{is_infinite_string}_{config.seed}.pt"
        test_set_path = f"test_sets/mnist_mixture_{is_infinite_string}_{config.seed}.pt"
        is_set_path = f"is_sets/mnist_mixture_{is_infinite_string}_{config.seed}.pt"
        if os.path.exists(val_set_path):
            logging.info(f"Removing existing validation set cache at {val_set_path}")
            os.remove(val_set_path)
        if os.path.exists(test_set_path):
            logging.info(f"Removing existing test set cache at {test_set_path}")
            os.remove(test_set_path)
        if os.path.exists(is_set_path):
            logging.info(f"Removing existing IS set cache at {is_set_path}")
            os.remove(is_set_path)

        eval_num_samples = 10000
        val_set = sample_tensor_dataset(eval_num_samples)
        other_val_set = sample_tensor_dataset(eval_num_samples)
        torch.save({"samples": val_set, "other_samples": other_val_set}, val_set_path)
        del val_set
        del other_val_set

        test_set = sample_tensor_dataset(eval_num_samples)
        other_test_set = sample_tensor_dataset(eval_num_samples)
        torch.save(
            {"samples": test_set, "other_samples": other_test_set}, test_set_path
        )
        del test_set
        del other_test_set

        is_num_samples = 100000
        is_set = sample_tensor_dataset(is_num_samples)
        torch.save({"samples": is_set}, is_set_path)
        del is_set

    return


def infinite_iterator(dist, batch_size, seed, process_index=0):
    torch.manual_seed(seed + process_index)
    while True:
        yield (dist.sample((batch_size,)), None)


def create_datasets(config, device=None, process_index=0, train=True):
    runtime_device = torch.device(config.device if device is None else device)
    train_num_workers = config.training.num_workers
    eval_num_workers = config.eval.num_workers
    train_pin_memory = True
    eval_pin_memory = config.eval.pin_memory

    def build_dataloader(
        dataset,
        batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    ):
        dataloader_kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        if persistent_workers and (num_workers > 0):
            dataloader_kwargs["persistent_workers"] = True
        return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    if config.data.name == "mnist_mixture":
        is_infinite = config.data.is_infinite
        is_infinite_string = "infinite" if is_infinite else "finite"
        mnist_dist, data_statistics = create_mnist_mixture_distribution(
            device=runtime_device
        )

        def build_eval_dataset(path):
            cached_set = torch.load(path, map_location="cpu")
            samples = cached_set["samples"]
            other_samples = cached_set.get("other_samples")
            if other_samples is None:
                return torch.utils.data.TensorDataset(samples)
            return torch.utils.data.TensorDataset(samples, other_samples)

        val_set = build_eval_dataset(
            f"val_sets/mnist_mixture_{is_infinite_string}_{config.seed}.pt"
        )
        val_dataloader = build_dataloader(
            val_set,
            batch_size=config.eval.batch_size,
            num_workers=eval_num_workers,
            pin_memory=eval_pin_memory,
        )

        test_set = build_eval_dataset(
            f"test_sets/mnist_mixture_{is_infinite_string}_{config.seed}.pt"
        )
        test_dataloader = build_dataloader(
            test_set,
            batch_size=config.eval.batch_size,
            num_workers=eval_num_workers,
            pin_memory=eval_pin_memory,
        )
        is_set = build_eval_dataset(
            f"is_sets/mnist_mixture_{is_infinite_string}_{config.seed}.pt"
        )
        is_dataloader = build_dataloader(
            is_set,
            batch_size=config.eval.batch_size,
            num_workers=eval_num_workers,
            pin_memory=eval_pin_memory,
        )

        if is_infinite:
            train_temp = infinite_iterator(
                mnist_dist,
                config.training.batch_size,
                seed=config.seed,
                process_index=process_index,
            )
        else:
            train_dataset = torch.utils.data.TensorDataset(
                torch.load(
                    f"train_sets/mnist_mixture_{is_infinite_string}_{config.seed}.pt",
                    map_location="cpu",
                )
            )

            train_temp = build_dataloader(
                train_dataset,
                batch_size=config.training.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=train_num_workers,
                pin_memory=train_pin_memory,
                persistent_workers=True,
            )

        return (
            train_temp,
            val_dataloader,
            test_dataloader,
            is_dataloader,
            is_infinite,
            mnist_dist,
            data_statistics,
        )

    elif config.data.name == "mnist":
        is_infinite = False

        dataset = MNIST(
            "datasets/mnist",
            train=True,
            transform=transforms.ToTensor(),
            download=True,
        )
        train_dataset = torch.utils.data.Subset(dataset, np.arange(50000))
        val_dataset = torch.utils.data.Subset(dataset, np.arange(50000, 60000))

        train_dataloader = build_dataloader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=train_num_workers,
            pin_memory=train_pin_memory,
            persistent_workers=True,
        )
        val_dataloader = build_dataloader(
            val_dataset,
            batch_size=config.eval.batch_size,
            num_workers=eval_num_workers,
            pin_memory=eval_pin_memory,
        )
        test_dataset = MNIST(
            "datasets/mnist",
            train=False,
            transform=transforms.ToTensor(),
            download=True,
        )
        test_dataloader = build_dataloader(
            test_dataset,
            batch_size=config.eval.batch_size,
            num_workers=eval_num_workers,
            pin_memory=eval_pin_memory,
        )

        dummy_data_statistics = {
            "means": torch.zeros((784,)),
            "std": 1.0,
        }

        return (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            None,
            is_infinite,
            None,
            dummy_data_statistics,
        )

    elif config.data.name == "imagenet64":
        is_infinite = False
        transform = _build_imagenet64_transform(config)
        val_subset_size = config.data.val_subset_size
        data_root = "../../imagenet64"
        dummy_data_statistics = {
            "means": torch.zeros((int(np.prod(config.data.shape)),)),
            "std": 1.0,
        }

        if not train:
            test_dataloaders = {}
            for set_name, transform in [
                (
                    "test_with_raw",
                    _build_eval_imagenet64_transform(config),
                ),
                (
                    "test_with_random_horizontal_flip",
                    _build_imagenet64_transform(config),
                ),
                (
                    "test_with_always_horizontal_flip",
                    _build_always_flip_eval_imagenet64_transform(config),
                ),
            ]:
                test_dataset = ImageNet64(
                    root=data_root,
                    train=False,
                    transform=transform,
                    download=False,
                )
                test_dataloaders[set_name] = build_dataloader(
                    test_dataset,
                    batch_size=config.eval.batch_size,
                    num_workers=eval_num_workers,
                    pin_memory=eval_pin_memory,
                )

            return (
                None,
                None,
                test_dataloaders,
                None,
                is_infinite,
                None,
                dummy_data_statistics,
            )

        full_train_dataset = ImageNet64(
            root=data_root,
            train=True,
            transform=transform,
            download=False,
        )

        val_subset_size = min(val_subset_size, len(full_train_dataset))
        split_index = len(full_train_dataset) - val_subset_size
        train_indices = np.arange(split_index)
        val_indices = np.arange(split_index, len(full_train_dataset))
        train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)

        official_eval_dataset = ImageNet64(
            root=data_root,
            train=False,
            transform=transform,
            download=False,
        )
        test_dataset = official_eval_dataset

        train_dataloader = build_dataloader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=train_num_workers,
            pin_memory=train_pin_memory,
            persistent_workers=True,
        )
        val_dataloader = build_dataloader(
            val_dataset,
            batch_size=config.eval.batch_size,
            num_workers=eval_num_workers,
            pin_memory=eval_pin_memory,
        )
        test_dataloader = build_dataloader(
            test_dataset,
            batch_size=config.eval.batch_size,
            num_workers=eval_num_workers,
            pin_memory=eval_pin_memory,
        )
        return (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            None,
            is_infinite,
            None,
            dummy_data_statistics,
        )

    raise ValueError(f"Unknown image dataset: {config.data.name}")
