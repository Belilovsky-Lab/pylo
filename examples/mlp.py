# Standard library imports
import argparse
import math
import os
import time
import io
from pickletools import optimize
import multiprocessing
import pickle

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import h5py
from PIL import Image
import tensorflow_datasets as tfds

import tensorflow as tf
# MuP imports
import torch
import random
import numpy as np
import os
import time

def set_torch_seed(seed = None, deterministic: bool = False):
    """
    Set all relevant seeds for PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
        deterministic (bool): If True, sets deterministic behavior for cuDNN (may slow down performance).
    """
    if seed is None:
        seed = int(time.time())

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU

    # Ensure deterministic behavior in CuDNN (at the cost of performance)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic  # If False, may slow down training

    # Ensuring reproducibility in distributed training
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" if deterministic else ""

    print(f"PyTorch seed set to {seed}. Deterministic: {deterministic}")

# Commented out imports
# import omegaconf

from pylo.optim import VeLO

def truncated_normal(lower, upper, shape=None, dtype=torch.float32, device=None):
    """
    Samples from a truncated normal distribution via inverse transform sampling.

    Args:
        lower (scalar or torch.Tensor): Lower bound(s) for truncation.
        upper (scalar or torch.Tensor): Upper bound(s) for truncation.
        shape (tuple or None): Desired output shape. If None, the broadcasted shape 
            of `lower` and `upper` is used.
        dtype (torch.dtype): The desired floating point type.
        device (torch.device or None): The device on which the tensor will be allocated.

    Returns:
        torch.Tensor: Samples from a truncated normal distribution with the given shape.
    """
    # Convert lower and upper to tensors (if they are not already)
    lower = torch.as_tensor(lower, dtype=dtype, device=device)
    upper = torch.as_tensor(upper, dtype=dtype, device=device)
    
    # If shape is not provided, use the broadcasted shape of lower and upper.
    if shape is None:
        shape = torch.broadcast_tensors(lower, upper)[0].shape
    else:
        # Optionally, you could add shape-checking logic here to ensure that
        # lower and upper are broadcastable to the provided shape.
        pass

    # Ensure that the dtype is a floating point type.
    if not torch.empty(0, dtype=dtype).is_floating_point():
        raise TypeError("truncated_normal only accepts floating point dtypes.")
    
    # Compute sqrt(2) as a tensor.
    sqrt2 = torch.tensor(np.sqrt(2), dtype=dtype, device=device)
    
    # Transform the truncation bounds using the error function.
    a = torch.erf(lower / sqrt2)
    b = torch.erf(upper / sqrt2)
    
    # Sample uniformly from the interval [a, b]. (The arithmetic here
    # broadcasts a and b to the desired shape.)
    u = a + (b - a) * torch.rand(shape, dtype=dtype, device=device)
    
    # Transform back using the inverse error function.
    out = sqrt2 * torch.erfinv(u)
    
    # To avoid any numerical issues, clamp the output so that it remains within
    # the open interval (lower, upper). Here we use torch.nextafter to compute the 
    # next representable floating point values:
    lower_bound = torch.nextafter(lower.detach(), torch.full_like(lower, float('inf')))
    upper_bound = torch.nextafter(upper.detach(), torch.full_like(upper, float('-inf')))
    out = torch.clamp(out, min=lower_bound, max=upper_bound)
    
    return out


class VarianceScaling:
    """Variance scaling initializer that adapts its scale to the shape of the initialized tensor.

    Args:
        scale (float): Scaling factor to multiply the variance by.
        mode (str): One of "fan_in", "fan_out", "fan_avg".
        distribution (str): One of "truncated_normal", "normal", or "uniform".
        fan_in_axes (Optional[Tuple[int]]): Optional sequence specifying the axes for fan-in calculation.
    """

    def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal', fan_in_axes=None):
        if scale < 0.0:
            raise ValueError('`scale` must be a positive float.')
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError(f'Invalid `mode` argument: {mode}')
        if distribution not in {'normal', 'truncated_normal', 'uniform'}:
            raise ValueError(f'Invalid `distribution` argument: {distribution}')
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.fan_in_axes = fan_in_axes

    def _compute_fans(self, shape):
        """Compute the fan-in and fan-out for the given shape."""
        dimensions = len(shape)
        if self.fan_in_axes is None:
            if dimensions == 2:  # For Linear layers
                fan_in, fan_out = shape[1], shape[0]
            else:  # For Conv layers
                receptive_field_size = math.prod(shape[2:])  # multiply all dimensions except first two
                fan_in = shape[1] * receptive_field_size
                fan_out = shape[0] * receptive_field_size
        else:
            # Custom fan-in based on specific axes
            fan_in = math.prod([shape[i] for i in self.fan_in_axes])
            fan_out = shape[0]
        return fan_in, fan_out

    def initialize(self, tensor):
        """Apply the initialization to the given tensor."""
        shape = tensor.shape
        fan_in, fan_out = self._compute_fans(shape)

        print("fan_in",fan_in)
        print("fan_out",fan_out)
        
        # Calculate the scale based on mode
        if self.mode == 'fan_in':
            scale = self.scale / max(1.0, fan_in)
        elif self.mode == 'fan_out':
            scale = self.scale / max(1.0, fan_out)
        else:  # fan_avg
            scale = self.scale / max(1.0, (fan_in + fan_out) / 2.0)

        if self.distribution == 'truncated_normal':
            stddev = math.sqrt(scale)
            return self._truncated_normal_(tensor, mean=0.0, std=stddev)

        elif self.distribution == 'normal':
            stddev = math.sqrt(scale)
            return torch.nn.init.normal_(tensor, mean=0.0, std=stddev)

        elif self.distribution == 'uniform':
            limit = math.sqrt(3.0 * scale)
            return torch.nn.init.uniform_(tensor, a=-limit, b=limit)

    @staticmethod
    def _truncated_normal_(tensor, mean=0.0, std=1.0):
        """Fill the tensor with values drawn from a truncated normal distribution."""
        with torch.no_grad():
            size = tensor.shape
            tensor.data.copy_(truncated_normal(lower=-2, upper=2, shape=size,))
            tensor.mul_(std).add_(mean)
            return tensor
            


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        width=128,
        num_classes=10,
        nonlin=F.relu,
        bias=True,
        log_activations=True,
    ):
        super(MLP, self).__init__()
        self.nonlin = nonlin
        self.log_activations = log_activations
        self.fc_1 = nn.Linear(in_channels, width, bias=bias)
        self.fc_2 = nn.Linear(width, width, bias=bias)
        self.fc_3 = nn.Linear(width, width, bias=bias)
        self.fc_4 = nn.Linear(width, num_classes, bias=bias)  # Standard Linear instead of MuReadout
        self.reset_parameters()

    def reset_parameters(self):
        ini = VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal')
        ini.initialize(self.fc_1.weight)
        ini.initialize(self.fc_2.weight)
        ini.initialize(self.fc_3.weight)
        ini.initialize(self.fc_3.weight)
        # nn.init.ones_(self.fc_1.weight)
        # nn.init.ones_(self.fc_2.weight)
        # nn.init.ones_(self.fc_3.weight)
        # nn.init.ones_(self.fc_4.weight)
        nn.init.zeros_(self.fc_1.bias)
        nn.init.zeros_(self.fc_2.bias)
        nn.init.zeros_(self.fc_3.bias)
        nn.init.zeros_(self.fc_4.bias)

    def forward(self, x):
        activations = {}  # Dictionary to store activation logs

        out = self.fc_1(x.flatten(1))
        if self.log_activations:
            with torch.no_grad():
                activations["layer_0_pre-act_l1"] = torch.mean(torch.abs(out)).item()
                activations["layer_0_pre-act"] = out.clone().detach()
        
        out = self.nonlin(out)
        if self.log_activations:
            with torch.no_grad():
                activations["layer_0_act_l1"] = torch.mean(torch.abs(out)).item()
                activations["layer_0_act"] = out.clone().detach()

        pre_out = self.fc_2(out)
        if self.log_activations:
            with torch.no_grad():
                activations["layer_1_pre-act_l1"] = torch.mean(torch.abs(pre_out)).item()
                activations["layer_1_pre-act"] = pre_out.clone().detach()
        
        out = self.nonlin(pre_out)
        if self.log_activations:
            with torch.no_grad():
                activations["layer_1_act_l1"] = torch.mean(torch.abs(out)).item()
                activations["layer_1_act"] = out.clone().detach()

        pre_out = self.fc_3(out)
        if self.log_activations:
            with torch.no_grad():
                activations["layer_2_pre-act_l1"] = torch.mean(torch.abs(pre_out)).item()
                activations["layer_2_pre-act"] = pre_out.clone().detach()
        
        out = self.nonlin(pre_out)
        if self.log_activations:
            with torch.no_grad():
                activations["layer_2_act_l1"] = torch.mean(torch.abs(out)).item()
                activations["layer_2_act"] = out.clone().detach()

        pre_out = self.fc_4(out)
        
        out = pre_out
        if self.log_activations:
            with torch.no_grad():
                activations["layer_3_logits_l1"] = torch.mean(torch.abs(out)).item()
                activations["layer_3_logits"] = out.clone().detach()
        
        return out, activations



def _cached_tfds_load(datasetname, split, batch_size):
    assert batch_size == -1
    return tfds.load(datasetname, split=split, batch_size=-1)
  

class Timer_dataset:
    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.func(*args, **kwargs)
        end_time = time.time()
        print(f"Executing {self.func.__name__} took {end_time - start_time:.4f} seconds.")
        return result


def process_batch(encoded_images):
    """Process a batch of encoded images into numpy arrays."""
    return [np.array(Image.open(io.BytesIO(img_data)).convert('RGB')) for img_data in encoded_images]

class H5Data:
    _instance = None

    @Timer_dataset
    def __new__(cls, h5_path, num_workers=8):
        if cls._instance is None:
            print("Creating the dataset instance")
            cls._instance = super(H5Data, cls).__new__(cls)
            
            # Read the encoded images and labels from the H5 file
            with h5py.File(h5_path, 'r') as file:
                encoded_images = file['encoded_images'][:]
                targets = file['targets'][:]
            
            # Determine the number of workers
            if num_workers is None:
                num_workers = multiprocessing.cpu_count()
            
            # Create batches of encoded images
            batch_size = len(encoded_images) // num_workers
            image_batches = [encoded_images[i:i + batch_size] for i in range(0, len(encoded_images), batch_size)]

            # Use multiprocessing to process the batches
            with multiprocessing.Pool(num_workers) as pool:
                image_arrays = pool.map(process_batch, image_batches)
            
            # Flatten the list of lists to a single list
            cls._instance.data = np.array([img for sublist in image_arrays for img in sublist])
            cls._instance.labels = np.squeeze(targets)

        return cls._instance

class NumpyImageDataset(Dataset):
    def __init__(self,transform=None):
        """
        Args:
            image_array (numpy.ndarray): The numpy array of images.
            label_array (numpy.ndarray): The numpy array of labels corresponding to the images.
        """
        # assert image_array.shape[0] == label_array.shape[0], "The number of images and labels must match."
        # self.split = split
        # self.h5_path = h5_path
        self.n_train = 1024933 #1281167
        self.n_val = 50000
        self.n_test = 100000
        # self.data = H5Data(h5_path=h5_path, num_workers=num_workers)
        
        # Convert numpy arrays to PyTorch tensors
        self.images = torch.from_numpy(H5Data._instance.data[:self.n_train]).float() / 255 #).permute(0, 3, 1, 2)
        self.labels = torch.from_numpy(H5Data._instance.labels[:self.n_train]).long()

        print(self.images.shape)
        print(self.labels.shape)
        self.transform = transform
    
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        # print('image',image.shape)
        if self.transform:
            image = self.transform(image.permute(2,1,0))
        
        # print(image.shape)
        return image, label




def _image_map_fn(cfg, batch):
  """Apply transformations + data aug to batch of data."""
  # batch is the entire tensor, with shape:
  # [batchsize, img width, img height, channels]
  batch = {k: v for k, v in batch.items()}
  if tuple(batch["image"].shape[1:3]) != cfg["image_size"]:
    batch["image"] = tf.image.resize(batch["image"], cfg["image_size"])

  if cfg["stack_channels"] != 1:
    assert batch["image"].shape[3] == 1, batch["image"].shape
    batch["image"] = tf.tile(batch["image"], (1, 1, 1, cfg["stack_channels"]))

  if cfg["aug_flip_left_right"]:
    batch["image"] = tf.image.random_flip_left_right(batch["image"])

  if cfg["aug_flip_up_down"]:
    batch["image"] = tf.image.random_flip_up_down(batch["image"])

  if cfg["normalize_mean"] is None:
    batch["image"] = tf.cast(batch["image"], tf.float32) / 255.
  else:
    assert cfg["normalize_std"] is not None
    image = tf.cast(batch["image"], tf.float32)
    image -= tf.constant(
        cfg["normalize_mean"], shape=[ 1, 1, 3], dtype=image.dtype)
    batch["image"] = image / tf.constant(
        cfg["normalize_std"], shape=[ 1, 1, 3], dtype=image.dtype)

  if cfg["convert_to_black_and_white"]:
    batch["image"] = tf.reduce_mean(batch["image"], axis=3, keepdims=True)

  batch["label"] = tf.cast(batch["label"], tf.int32)
  
  # return hk.data_structures.to_immutable_dict({
  #     "image": jax.device_put(batch["image"],device=jax.devices('gpu')[0]),
  #     "label": jax.device_put(batch["label"],device=jax.devices('gpu')[0])
  # })
  
  return batch["image"], batch["label"]
  
  
#   hk.data_structures.to_immutable_dict({
#       "image": batch["image"],
#       "label": batch["label"]
#   })


def generate_data(num_samples=10000, num_classes=1000, img_shape=(64, 64, 3), seed=42):
    np.random.seed(seed)
    
    # Generate class labels
    labels = np.random.randint(0, num_classes, size=num_samples).astype(np.int64)
    
    # Generate class-specific means (each class has a unique mean)
    class_means = np.linspace(0, 255, num_classes).reshape(-1, 1, 1, 1)
    
    # Generate images with the corresponding mean for each class
    images = np.zeros((num_samples, *img_shape), dtype=np.uint8)
    for i in range(num_samples):
        class_mean = class_means[labels[i]]
        images[i] = np.clip(np.random.normal(loc=class_mean, scale=30, size=img_shape), 0, 255).astype(np.float32)
    
    return labels, images


class RandomNumpyImageDataset(Dataset):
    def __init__(self,transform=None):
        """
        Args:
            image_array (numpy.ndarray): The numpy array of images.
            label_array (numpy.ndarray): The numpy array of labels corresponding to the images.
        """
        self.transform = transform
        self.labels, self.images =  generate_data()
        print(self.images.shape)
        print(self.labels.shape)
    
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        # print('image',image.shape)
        # if self.transform:
        #     image = self.transform(image.permute(2,1,0))
        
        # print(image.shape)
        return self.transform(image), label


class CNumpyImageDataset(Dataset):
    def __init__(self,transform=None):
        """
        Args:
            image_array (numpy.ndarray): The numpy array of images.
            label_array (numpy.ndarray): The numpy array of labels corresponding to the images.
        """

        
        dataset = _cached_tfds_load(datasetname='cifar10', split='train', batch_size=-1)
        

        cfg = {
            "batch_size": {'train':4096},
            "image_size": (32,32),
            "stack_channels": 1,
            "prefetch_batches": {'train':20},
            "aug_flip_left_right": False,
            "aug_flip_up_down": False,
            "normalize_mean": None,
            "normalize_std": None,
            "convert_to_black_and_white": False,
        }
        images, labels = tfds.as_numpy(_image_map_fn(cfg, dataset))
        
        # Convert numpy arrays to PyTorch tensors
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()

        print(self.images.shape)
        print(self.labels.shape)
        self.transform = transform
    
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        # print('image',image.shape)
        # print('label',label.shape)

        if self.transform:
            image = self.transform(image.permute(2,1,0))
        
        # print(image.shape)
        return image, label



def get_dataset(args):

    if args.dataset_name == 'imagenet':
        H5Data(h5_path=f'/home/mila/p/paul.janson/data/imagenet_{args.image_size}x{args.image_size}x3_JPEG.h5')

        normalize_mean=(0.485, 0.456, 0.406)
        normalize_std=(0.229, 0.224, 0.225)
        
        transform = transforms.Compose([
            transforms.Normalize(normalize_mean, normalize_std)
        ])

        trainset = NumpyImageDataset(transform)

        random_sampler = RandomSampler(
            trainset, 
            replacement=True,
            num_samples=args.num_iters * 4096 * args.num_trials
        )

        train_loader = torch.utils.data.DataLoader(
            trainset, 
            sampler=random_sampler, 
            batch_size=args.batch_size,
            shuffle=not args.no_shuffle, 
            drop_last=True,
            num_workers=8,
            prefetch_factor=10,
            pin_memory=True,
            persistent_workers=False
        )
        test_loader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=4096, 
            shuffle=False, 
            drop_last=True,
            num_workers=8,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=False
        )

    elif args.dataset_name == 'cifar10':


        mean_cifar10 = (0.49139968, 0.4821584,  0.44653094)  # CIFAR-10 dataset mean (per channel)
        std_cifar10 = (0.24703221, 0.24348514, 0.26158786)
        transform = transforms.Compose(
                [
                transforms.Normalize(mean=mean_cifar10, std=std_cifar10)]
        )

        trainset = datasets.CIFAR10(root=args.data_dir, train=True,
                                                download=True, transform=transform)
        trainset = CNumpyImageDataset(transform)

        random_sampler = RandomSampler(
            trainset, 
            replacement=True,
            num_samples=args.num_iters * 4096 * args.num_trials
        )
        train_loader = torch.utils.data.DataLoader(trainset, 
                                                   batch_size=args.batch_size,
                                                   sampler=random_sampler, 
                                                   prefetch_factor=10,
                                                    shuffle=not args.no_shuffle, 
                                                    num_workers=8,
                                                    pin_memory=True)

        testset = datasets.CIFAR10(root=args.data_dir, train=False, 
                                            download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=2)    
    elif args.dataset_name == 'random':
        
        transform = transforms.Compose(
                [transforms.ToTensor(),]
        )
        trainset = RandomNumpyImageDataset(transform)

        random_sampler = RandomSampler(
            trainset, 
            replacement=True,
            num_samples=args.num_iters * args.batch_size * args.num_trials
        )

        train_loader = torch.utils.data.DataLoader(
            trainset, 
            sampler=random_sampler, 
            batch_size=args.batch_size,
            shuffle=not args.no_shuffle, 
            drop_last=True,
            num_workers=8,
            prefetch_factor=10,
            pin_memory=True,
            persistent_workers=False
        )
        test_loader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            drop_last=True,
            num_workers=8,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=False
        )
    else:
        raise ValueError('Invalid dataset name')
    
    return train_loader, test_loader

def test(
    args, model, device, test_loader, evalmode=True, criterion=F.cross_entropy
):
    if evalmode:
        model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.view(data.size(0), -1))
            test_loss += criterion(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return test_loss, correct / len(test_loader.dataset)


if __name__ == '__main__':
    
    logs = [] 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(image.shape,label.shape)
    # exit(0)

    class args(object):
        pass

    args.lr = 1
    args.momentum = 0.9
    args.data_dir = os.environ["DATA_PATH"]
    args.batch_size = 4096
    args.no_shuffle = True
    args.epochs = 100
    args.log_interval = 1
    args.num_iters = 5000
    args.width = 128
    args.wandb_checkpoint_id = 'eb-lab/mup-meta-training/woz3g9l0'
    args.optimizer = 'mup_small_fc_mlp'
    args.num_trials = 3
    args.image_size = 64
    args.dataset_name = 'imagenet'
    args.log_norm = True
    args.use_wandb = True
    args.use_bf16 = False
    args.num_classes = 1000

    IS = args.image_size
    args.group_name = f"standard_fc_mlp32_['mlp-w{args.width}-d3_{args.dataset_name}-{IS}x{IS}x3']_velo_torch_imagenet"





    MODEL_TESTING = False
    if MODEL_TESTING:
        input_width = args.image_size * args.image_size * 3
        model = MLP(input_width, width=args.width, num_classes=args.num_classes).to(device)

        #set seed before re-init parameters
        set_torch_seed()
        model.reset_parameters()

        # set the biases to have base_width = 1 
        for n,p in model.named_parameters():
            if p.infshape.main != None:
                print(n, p.infshape.main.width_mult(),p.shape,p.infshape.ninf())
                with torch.no_grad():
                    print("{} {} {}".format(p.std(),p.mean(),p.infshape))
                # print(,p.bias.shape)
            # if 'bias' in n:
            #     if p.infshape.main:
            #         p.infshape.main.base_dim = width
        exit(0)
        

    # set transforms to match jax
    transform = transforms.Compose(
            [transforms.ToTensor(),]
    )
    train_loader, test_loader = get_dataset(args)
    train_loader = iter(train_loader)



    if not args.use_bf16:
        torch.set_float32_matmul_precision('high')



    

    for _ in range(args.num_trials):

        if args.use_wandb:
            run = wandb.init(
                            project='mup-meta-testing',
                            config=dict(wandb_checkpoint_id='eb-lab/mup-meta-training/woz3g9l0',
                                        optimizer='mup_small_fc_mlp'),
                            group=args.group_name
                        )

        input_width = args.image_size * args.image_size * 3
        model = MLP(input_width, width=args.width, num_classes=args.num_classes).to(device)

        #set seed before re-init parameters
        set_torch_seed()
        model.reset_parameters()
        
        # Compile the model
        # if torch.cuda.is_available():
        #     model = torch.compile(model)


        # optimizer = MuSGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        #optimizer = MuLO_naive(model.parameters(), lr=args.lr)
        #optimizer = MuLO_CUDA(model.parameters(), lr=args.lr)
        optimizer = VeLO(model.parameters(), lr=args.lr,num_steps=args.num_iters)
        print("Optimizer",optimizer)


        
        model.train()
        if torch.cuda.is_available() and args.use_bf16:
            model = model.to(torch.bfloat16)

        pbar = tqdm(
            range(args.num_iters),
            initial=0,
            total=args.num_iters,
            ascii=True,
            desc="Inner Loop",
        )
        for batch_idx in pbar:
            data, target = next(train_loader)
            if torch.cuda.is_available() and args.use_bf16:

                model = model.to(torch.bfloat16)
                data = data.to(torch.bfloat16)
                
            # if torch.cuda.is_available():
            #     data = data.pin_memory().to(device, non_blocking=True)
            #     target = target.pin_memory().to(device, non_blocking=True)
            # else:
            #     data, target = data.to(device), target.to(device)


            data, target = data.to(device), target.to(device)
            # Flatten data once
            data_flattened = data.view(data.size(0), -1)
            optimizer.zero_grad()
            output, activations = model(data_flattened)
            loss = F.cross_entropy(output, target)
            loss.backward()

            # Combine operations to reduce memory access
            with torch.no_grad():
                pred = output.argmax(dim=1, keepdim=True)
                accuracy = pred.eq(target.view_as(pred)).sum().item()/data.size(0)

            temp = {
                "iteration": batch_idx,
                "train loss": loss.item(),
                "train_acc": accuracy,
            }

            
            if args.use_bf16:
                #cast to FP32 so kernel works
                model = model.to(torch.float32)

            if args.log_norm:
                with torch.no_grad():
                    for n,p in model.named_parameters():
                        temp[f"{n}_norm"] = torch.norm(p).item()

            if isinstance(optimizer, VeLO):
                optimizer.step(loss)
            else:
                optimizer.step()

            # test interval
            # if batch_idx % 50 == 0:
            #     test_loss, test_acc = test(args, model, device, test_loader)
            #     model.train()

            #     temp.update(
            #         {"test_loss":test_loss,
            #         "test_acc":test_acc}
            #     )

            pbar.set_postfix(temp)
            

            if args.use_wandb:
                temp.update(activations)
                run.log(temp)

        if args.use_wandb:
            wandb.finish()

        # del train_loader
        # del test_loader

    exit(0)
