# DiffESM
Using diffusion to approximate Earth System Model (ESM) in the temperature and precipitation sphere. Generative modeling of this kind can be used for identification of future climate anomalies such as heat waves or dry spells.

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
Due to the necessity of large datasets for training diffusion model, we run a pre-processing script to chunk the dataset for multi-process loading and save it to disk. This is done through the following commands:

1. Gather all of the data into a single directory. They should all be stored in .nc format. 

2. Prepare a JSON file describing the structure of your dataset and the variables that comprise it. See below for an example JSON:

```json
{
   "load_dir" : "/path/to/data_directory/",
    "realizations" : {

    "r1" : {
    "pr" : [
        "pr_day_IPSL-CM5A-LR_historical_r1i1p1_18500101-19491231.nc",
        "pr_day_IPSL-CM5A-LR_historical_r1i1p1_19500101-20051231.nc",
        "pr_day_IPSL-CM5A-LR_rcp85_r1i1p1_20060101-22051231.nc"
    ],
    "tas" : [
        "tas_day_IPSL-CM5A-LR_historical_r1i1p1_18500101-19491231.nc",
        "tas_day_IPSL-CM5A-LR_historical_r1i1p1_19500101-20051231.nc",
        "tas_day_IPSL-CM5A-LR_rcp85_r1i1p1_20060101-22051231.nc"
    ]
},
"r2" : {
    "pr" : [
        "pr_day_IPSL-CM5A-LR_historical_r2i1p1_18500101-18991231.nc",
        "pr_day_IPSL-CM5A-LR_historical_r2i1p1_19000101-19491231.nc",
        "pr_day_IPSL-CM5A-LR_historical_r2i1p1_19500101-19991231.nc",
        "pr_day_IPSL-CM5A-LR_historical_r2i1p1_20000101-20051231.nc",
        "pr_day_IPSL-CM5A-LR_rcp85_r2i1p1_20060101-21001231.nc"
    ],
    "tas" : [
        "tas_day_IPSL-CM5A-LR_historical_r2i1p1_18500101-18991231.nc",
        "tas_day_IPSL-CM5A-LR_historical_r2i1p1_19000101-19491231.nc",
        "tas_day_IPSL-CM5A-LR_historical_r2i1p1_19500101-19991231.nc",
        "tas_day_IPSL-CM5A-LR_historical_r2i1p1_20000101-20051231.nc",
        "tas_day_IPSL-CM5A-LR_rcp85_r2i1p1_20060101-21001231.nc"
    ]
},
"r3" : {
    "pr" : [
        "pr_day_IPSL-CM5A-LR_historical_r3i1p1_18500101-19491231.nc",
        "pr_day_IPSL-CM5A-LR_historical_r3i1p1_19500101-20051231.nc",
        "pr_day_IPSL-CM5A-LR_rcp85_r3i1p1_20060101-21001231.nc"
    ],
    "tas" : [
        "tas_day_IPSL-CM5A-LR_historical_r3i1p1_18500101-19491231.nc",
        "tas_day_IPSL-CM5A-LR_historical_r3i1p1_19500101-20051231.nc",
        "tas_day_IPSL-CM5A-LR_rcp85_r3i1p1_20060101-21001231.nc"
    ]
},
"r4" : {
    "pr" : [
        "pr_day_IPSL-CM5A-LR_historical_r4i1p1_18500101-18991231.nc",
        "pr_day_IPSL-CM5A-LR_historical_r4i1p1_19000101-19491231.nc",
        "pr_day_IPSL-CM5A-LR_historical_r4i1p1_19500101-19991231.nc",
        "pr_day_IPSL-CM5A-LR_historical_r4i1p1_20000101-20051231.nc",
        "pr_day_IPSL-CM5A-LR_rcp85_r4i1p1_20060101-21001231.nc"
    ],
    "tas" : [
        "tas_day_IPSL-CM5A-LR_historical_r4i1p1_18500101-18991231.nc",
        "tas_day_IPSL-CM5A-LR_historical_r4i1p1_19000101-19491231.nc",
        "tas_day_IPSL-CM5A-LR_historical_r4i1p1_19500101-19991231.nc",
        "tas_day_IPSL-CM5A-LR_historical_r4i1p1_20000101-20051231.nc",
        "tas_day_IPSL-CM5A-LR_rcp85_r4i1p1_20060101-21001231.nc"
    ]
},
"r5" : {
    "pr" : [
        "pr_day_IPSL-CM5A-LR_historical_r5i1p1_18500101-19491231.nc",
        "pr_day_IPSL-CM5A-LR_historical_r5i1p1_19500101-20051231.nc",
        "pr_day_IPSL-CM5A-LR_rcp85_r3i1p1_20060101-21001231.nc"
    ],
    "tas" : [
        "tas_day_IPSL-CM5A-LR_historical_r5i1p1_18500101-19491231.nc",
        "tas_day_IPSL-CM5A-LR_historical_r5i1p1_19500101-20051231.nc",
        "tas_day_IPSL-CM5A-LR_rcp85_r3i1p1_20060101-21001231.nc"
    ]
},

"r6" : {
    "pr" : [
        "pr_day_IPSL-CM5A-LR_historical_r6i1p1_18500101-19491231.nc",
        "pr_day_IPSL-CM5A-LR_historical_r6i1p1_19500101-20051231.nc",
        "pr_day_IPSL-CM5A-LR_rcp85_r4i1p1_20060101-21001231.nc"
    ],
    "tas" : [
        "tas_day_IPSL-CM5A-LR_historical_r6i1p1_18500101-19491231.nc",
        "tas_day_IPSL-CM5A-LR_historical_r6i1p1_19500101-20051231.nc",
        "tas_day_IPSL-CM5A-LR_rcp85_r4i1p1_20060101-21001231.nc"
    ]
   }
}
}
```

3. Save the JSON file to a directory. The directory should be in the form of: `/{leading path to directory}/{name of ESM (IPSL)}/{name of scenario (rcp85)}/data.json`

4. In `configs/paths/default.yaml` (or other config file in the `paths` directory), specify the leading path to all the JSON files under `json_data_dir` and specify the leading path to the directory where you want to save the processed data under `data_dir`.

5. Specify the ESM, Scenario, start and end year of your dataset, and the number of chunks to split it into in `configs/prepare_data.yaml`. Then run `make prepare_data` to process the data. Note, this may take up to an hour with large datasets.



  
