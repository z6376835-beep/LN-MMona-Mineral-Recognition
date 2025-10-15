LN-MMona:A joint fine-tuning method for mineral image recognition

:wrench: Installation
1. Create and activate a conda environment:
conda create -n lnmmona python=3.8 -y
conda activate lnmmona

2. Install dependencies:
pip install -r requirements.txt

3. Install the project:
git clone <repository_url>
cd LN-MMona
pip install -e .

:floppy_disk: Prepare Datasets

Ensure that your dataset is organized in the ImageFolder format as described in the repository. You can download the datasets from the following links:

Minet V2: Download PDF - Minet V2 Dataset

Minerals Identification Dataset: Download from Kaggle

:rocket: Training and Evaluation
Start Training and Evaluation

To train and evaluate the model, simply run the following command:

python train.py

What this command will do:

Train the model on your dataset.

Evaluate the model's performance on the validation and test sets.

Log key metrics such as accuracy, F1 score, and others during the process.

:gear: Hyperparameters

You can adjust the following training hyperparameters in the train.py file to better suit your needs:

batch_size: The size of the training batches.

learning_rate: The learning rate for the optimizer.

num_epochs: Number of epochs for training.

num_runs: Number of runs to train.

log_file: Path to save training logs.

:page_facing_up: License

This project is licensed under the MIT License. See the LICENSE
 file for details.