# DiffESM
Diffusion model software to emulate Earth System Models (ESMs) for daily temperature and precipitation. This software is capable of generating new daily precipitation or temperature data for previously unseen scenarios, with many potential applications (e.g., estimating or characterizing extreme weather phenomena such heat waves or dry spells under hypothetical future climate scenarios).

## Setup Instructions

1. We use weights and biases for logging. To use it, you will need to make an account: [https://wandb.ai/site](https://wandb.ai/site)

2. To set up your environment using Conda, follow these steps:

   a. First, ensure you have Conda installed on your system. If not, download and install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual).

   b. Clone the repository to your local machine:
      ```bash
      git clone https://github.com/your-username/your-repo-name.git
      cd your-repo-name
      ```

   c. Create a Conda environment using the `environment.yml` file provided in the repository:
      ```bash
      conda env create -f environment.yml
      ```

   d. Activate the newly created environment:
      ```bash
      conda activate diffesm
      ```

   e. After activating the environment, you can proceed with the rest of the setup.

## Preparing the Data
To train the diffusion model, we first have to preprocess the data into a format that the training script is expecting. Follow these steps to preprocess and organize your data:

### Step 1: Consolidate Dataset
Collect all data files into a single directory. Ensure each file is in `.nc` format.

### Step 2: Create Dataset Description
Develop a JSON file to describe your dataset's structure and its variables. This file should outline at least three realizations for each of the training, validation, and testing sets.

Example JSON structure:
```json
{
   "load_dir" : "/path/to/data_directory/",
    "realizations" : {
        // Example of realizations for precipitation (pr) and temperature (tas)
        // under different scenarios and time frames
        "r1" : {
            "pr" : ["file1_1850_1950.nc", "file2_1950_2100.nc", ...],
            "tas" : ["file3_1850_2006.nc", "file4_2006_2100.nc", ...]
        },
        "r2" : {...},
        "r3" : {...}
    }
}
```
### Step 3: Save JSON file
Store the JSON file in a structured directory format:
`/{path_to_directory}/{ESM_name}/{scenario_name}/data.json`

### Step 4: Update Configuration Paths
Modify `configs/paths/default.yaml` (or an alternative configuration file in the paths directory) to include:

- **json_data_dir**: The leading path to the JSON files.
- **data_dir**: The path to the directory where processed data will be stored.

### Step 5: Run Preprocessing Script
In `configs/prepare_data.yaml` specify:
- The Earth System Model (IPSL, CESM, etcm...)
- Scenario (rcp85, rcp45, etc...)
- Dataset's start and end years
- Number of chunks for data splitting

Finally run `make prepare_data` to start processing. Note: This may take up to an hour for large datasets, but will only need to be run once



## Training the Model

After your data is ready, follow these steps to train the diffusion model, which aims to approximate the Earth System Model (ESM):

### Configuration Setup
The `configs` directory contains all necessary configuration files for training. `configs/train.yaml` selects the default configuration for the following options

- **Model Architecture:** Located in `config/model/`. These files defines the structure of the diffusion model.
- **Scheduler:** Found in `config/scheduler/`. It manages the diffusion scheduler we use and defines the noising and denoising process.
- **Dataset Configuration:** Specified in `config/data/`. It details what ESM and variables you want to use for your dataset.
- **Training Hyperparameters:** Located in `config/trainer/`. This file includes settings like batch size, learning rate, and other critical parameters for training.

### Hyperparameter Customization
Adjust the hyperparameters in the configuration files to suit your specific training requirements.

### Training Script Configuration
Use the `scripts/train.sh` script to set the number of GPUs for training.

### Start Training
Once all configurations are set, initiate the training process by running the command:
```bash
make train
```

This will start the model training based on your specified configurations.

## Evaluation

The evaluation process involves generating and comparing 20 years of daily data to assess the model's performance. This is done in two main steps:

### Step 1: Generating Validation and Test Sets
- **Configure Data Generation:** Use the `generate.yaml` file to specify the type of data you want to generate (ESM, scenario, start/end years, and validation/test). This file is crucial for defining the parameters of your data generation process.
- **Initial Generation:** First, generate the validation and test sets using the original Earth System Model (ESM) data. This step does not involve the trained model but relies on the ESM data to create baseline datasets.

### Step 2: Generating Data with Trained Model
- **Run Model Generation:** After creating the baseline datasets, run the same generation process, this time using your trained model. This will allow you to produce data that reflects the model's capabilities.
- **Saving Generated Data:** The output from this process will be automatically saved to the directory specified as `save_dir` in the "paths" configuration file.

### Executing the Generation Script
To initiate the data generation process for both steps, execute the following command:
```bash
make generate
```

**Additional Notes:**

- **Process Configuration:** The number of processes used during generation is set in the `scripts/gen_sample.sh` script.
- **Time Consideration:** Depending on your hardware setup, the generation process may take several minutes.


## Visualization
Finally, you are ready to visualize your results! The bulk of visualization is performed in the `notebooks/data-viz.ipynb` notebook. The configuration for the data vizualization is specified in `configs/data_viz.yaml`. Currently, the notebook only supports vizualizations for temperature and precipitation, although other variables can be added in the future.
