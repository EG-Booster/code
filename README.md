# EG-Booster: Explanation-Guided Booster of ML Evasion Attacks
This repository contains the source code accompanying our paper EG-Booster: Explanation-Guided Booster of ML Evasion Attacks.

## Used system
- Linux
- 6 vCPUs, 18.5 GB memory
- GPU: 1 x NVIDIA Tesla K80


### Downloading Repo
```$ git clone https://github.com/EG-Booster/code.git ```



### CIFAR10
It is highly recommended to create a new separate python3 environment:

```$ python3 -m venv ./EG-CIFAR10-env```

```$ source EG-CIFAR10-env/bin/activate```

```$ cd code/CIFAR10```

```$ pip install -r requirements.txt```

```$ python CIFAR10.py```

Note: to avoid any memory-related errors please adjust ```batch_size = 128``` and ```num_workers = 2``` in the configuration area of ```CIFAR10.py```, according to your system's specs.
