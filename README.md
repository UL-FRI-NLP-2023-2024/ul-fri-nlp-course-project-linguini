# Natural language processing course 2023/24: `Cross-Lingual Question Generation (Project 3)`

## Overview
This project focuses on expanding the capabilities of the Doc2Query approach by utilizing a T5 model. The primary objective is to explore the model's effectiveness and the quality of questions generated across different languages, identifying both the challenges and opportunities that arise when applying such models in a cross-linguistic context. As a specific case study, the project will involve fine-tuning the system on Slovenian datasets to evaluate its performance and output quality in a less commonly used language setting.
## Objectives
Todo
## Usage
### Getting started
This guide provides detailed instructions on how to setup the project environment using Singularity containers. Follow the steps below to build your environment from scratch.
#### Step 1: Create Container Directory
Create a folder for your containers in your home directory and move into the newly created folder.
```
mkdir containers
cd containers
```
#### Step 2: First layer of multi-layer container installation
Either copy file "base_build.def" provided in our repository into folder "containers" or make a new file.
```
nano base_build.def
```
The file needs to look like this:
```
Bootstrap: docker
From: nvidia/cuda:12.4.1-devel-ubuntu22.04

%post
    apt-get update && apt-get install -y python3 python3-pip git gcc
    apt-get clean && rm -rf /var/lib/apt/lists/*
    pip3 install --upgrade pip

```
Finally we build the container image with command from the 'containers' folder:
```
singularity build base_build.sif base_build.def
```
#### Step 3: Second and Third layers of container installation
Similarly, copy or create files "intermediate.def" which should look like this:
```
Bootstrap: localimage
From: base_build.sif

%post
    pip3 install numpy pandas scikit-learn trl transformers accelerate
    pip3 install 'git+https://github.com/huggingface/peft.git'
    pip3 install datasets bitsandbytes langchain sentence-transformers beautifulsoup4
```
and "final.def" which should look like this:
```
Bootstrap: localimage
From: intermediate.sif

%post
    # Custom installation scripts if necessary

%files
    # Transfer necessary files

%environment
    export PATH=/path/to/your/applications:$PATH

%runscript
    echo "Running script $*"
    exec python3 "$@"
```
After both files are set. Run following commands:
```
singularity build intermediate.sif intermediate.def
singularity build final.sif final.def

```
Finally, leave container directory with
```
cd ..
```
#### Step 4: Cloning repository
Run command:
```
git clone https://github.com/UL-FRI-NLP-2023-2024/ul-fri-nlp-course-project-linguini.git
```
#### Step 5: Creating SLURM job script
Create a script to run the project using SLURM.
```
nano test_run.sh
```
And copy the following into the file (or just use the script in our repository).
```
#!/bin/sh
#SBATCH --job-name=lora
#SBATCH --output=lora.log
#SBATCH --error=loraerr.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem=64G

srun singularity exec --nv ./containers/final.sif python3 "ul-fri-nlp-course-project-linguini/test.py"
```
#### Step 5: Submit job to SLURM
Finally run command:
```
sbatch test_run.sh
```
### Using the model
Todo
