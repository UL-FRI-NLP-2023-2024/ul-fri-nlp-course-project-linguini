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
Create new folder for logs:
```
mkdir logs
```

Create a script to run the project using SLURM.
```
nano test_run.sh
```
And copy the following into the file (or just use the script in our repository).
```
#!/bin/sh
#SBATCH --job-name=lora
#SBATCH --output=logs/lora.log
#SBATCH --error=logs/loraerr.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem=64G

srun singularity exec --nv ./containers/final.sif python3 "ul-fri-nlp-course-project-linguini/test.py"
```
#### Step 6: Submit job to SLURM
Finally run command:
```
sbatch test_run.sh
```
### Training the Doc2Query model

For training the Doc2Query model we can use the same container installation as before, we only need to create a new python training script and new shell script.

##### Step 1: Create python training file and running script

For the training file, you can copy the contents of the [finetuning.py](./Code/finetuning.py) into your file.

Then you need to create a new script to run the training python file:

Example if you use the container setup:

```
#!/bin/sh
#SBATCH --job-name=lora
#SBATCH --output=seq.log
#SBATCH --error=seq.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem=80G
#SBATCH --cpus-per-task=20
srun singularity exec --nv ./containers/final.sif python3  ul-fri-nlp-course-project-linguini/finetuning.py
```
#### Step 2: Submit job

To submit the job you do the same as in the [test example](#step-6-submit-job-to-slurm), just change the command to run your own running script.

### Querying the Doc2Query model

After the job you submitted before is finished, we created a new python file, which for each paragraph, generates 10 different queries (5 with beam search, 5 with top k random sampling).

To run this file, follow the same procedure as before. Use or copy the contents of the [Querying.py](./Code/queryingmodel.py) file provided in the repository. 


### Training the SloT5-small model

When training the model, remember to change the result destinations in the python files, so you don't overwrite your previous work and models.

#### Step 1: Training file
For training the SloT5 model we still use the same container installation as before, now we only need to modify the [training python file](./Code/finetuning.py) file to use this new model, and we also need to add a new parameter in the training arguments:
```
bf16=True
```
The training will be faster this way. You can also just use the [modified training file](./Code/finetuning_sloT5.py) provided in the project.

#### Step 2: Running script

Create a new running script the same way as [before](#step-1-create-python-training-file-and-running-script), be wary to choose the correct python file.


#### Step 3: Submit job

To submit the job you do the same as in the [test example](#step-6-submit-job-to-slurm), just change the command to run your own running script.

### Querying the SloT5-small model

Querying here is done the same way as in the Doc2query model. Be careful to change which model you use and where the queries are saved, so you don't overwrite your previous training files.

To run this file, follow the same procedure as before. Use or copy the contents of the [Querying.py](./Code/queryingmodel.py) file provided in the repository. 






