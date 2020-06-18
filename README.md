# Covid-19-Pneumonia
Pneumonia segmentation model for CT scans. To be used in the context of covid-19 studies.

## Setup: instructions for getting setup on the `prp-gpu-1.t2.ucsd.edu` machine
1. Add the following lines to `~/.bashrc`:
```
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/xilinx/scratch/software/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/xilinx/scratch/software/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/xilinx/scratch/software/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/xilinx/scratch/software/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```
2. `mkdir ~/.conda`
3. Create a file `~/.conda/environments.txt` and add these lines:
```
/xilinx/scratch/software/miniconda3
/xilinx/scratch/software/miniconda3/envs/tensorflow
```
4. `source ~/.bashrc`
5. `conda activate tensorflow`
