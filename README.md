# Covid-19-Pneumonia
Pneumonia segmentation model for CT scans. To be used in the context of covid-19 studies.

# Setup: instructions for getting setup on the `prp-gpu-1.t2.ucsd.edu` machine (thanks to Javier Duarte)
1. Install miniconda: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
`bash Miniconda3-latest-Linux-x86_64.sh`
2. Create an environment: `conda create -n tensorflow python=3.6` (this just creates an environment called ``tensorflow'')
3. Enter environment: `conda activate tensorflow`
4. Install tensorflow: `conda install -c anaconda tensorflow-gpu`
5. Make sure tensorflow is installed properly and can see the GPUs. In python interpreter
    - `import tensorflow as tf`
    - `print(tf.__version__)`
    - `print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))`
6. Install other necessary python packages via `pip`: 
