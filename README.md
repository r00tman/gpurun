# gpurun
Find free gpus and run stuff from argv on them

## Usage
Install dependencies: ```pip3 install gpustat```

Example usage: ```./gpurun.py -n 2 env```

That one tries to find 2 free gpus and then runs ```env``` with ```CUDA_VISIBLE_DEVICES``` properly set
