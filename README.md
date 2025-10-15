# LN-MMona: A Joint Fine-Tuning Method for Mineral Image Recognition

(flowchart.jpg)

## :wrench: Installation

### 1. Create and activate a Conda environment:
```bash
conda create -n lnmmona python=3.8 -y
conda activate lnmmona
````

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Install the project:

```bash
git clone https://github.com/z6376835-beep/LN-MMona-Mineral-Recognition.git
cd LN-MMona
pip install -e .
```

---

## :floppy_disk: Prepare Datasets

Ensure your dataset is organized in the **ImageFolder** format, as described in the repository.

You can download the datasets from the following links:

* **Minet V2**: [Download PDF - Minet V2 Dataset](#)
* **Minerals Identification Dataset**: [Download from Kaggle](https://www.kaggle.com/datasets)

---

## :rocket: Training and Evaluation

To train and evaluate the model, run the following command:

```bash
python train.py
```

### What this command does:

* **Train the model** on your dataset.
* **Evaluate the model's performance** on the validation and test sets.
* **Log key metrics** such as accuracy, F1 score, and others during the process.
## :page_facing_up: License

This project is licensed under the **MIT License**. See the [LICENSE](#) file for details.

