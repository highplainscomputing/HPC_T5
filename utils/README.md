# T5 Utility Functions for NLP Tasks

This repository contains a set of utility functions designed to streamline the process of working with the T5 model for various natural language processing (NLP) tasks. The functions provided aim to simplify tasks such as data preparation, model training, inference, and metric computation.

## Installation

Clone this repository and install the required dependencies:

```
git clone https://github.com/your-username/your-repository.git
cd your-repository
pip install -r requirements.txt
```
## Usage

### 1. Tokenizer and Metrics Loading

The **tokenizer_and_metrics** function loads the T5 tokenizer and metrics required for evaluation.

```
from t5_utils import tokenizer_and_metrics

tokenizer, metrics = tokenizer_and_metrics(MODEL, METRICS)
```

### 2. Loading Data

The **get_data** function(helping function) loads data either from Hugging Face datasets or from pandas DataFrame.

```
from t5_utils import get_data

# Load data from Hugging Face dataset
train_data = get_data("dataset_name", split="train")

# Load data from pandas DataFrame
train_data = get_data(data_frame, split="train")
```

### 3. Data Preparation

The **prepare_data** function converts data into the T5 required format.

```
from t5_utils import prepare_data

train_data = prepare_data(train_data, split="train", Q_col="question", A_col="answer", prefix="qa")

```

### 4. Identify Max Lengths

The **identify_max_lengths** function helps identify the maximum input and target lengths to assist the tokenizer.

```
from t5_utils import identify_max_lengths

max_input_length, max_target_length = identify_max_lengths(train_data, eval_data, model = "google/flan-t5-base")

```

### 5. Preprocessing Data

The preprocessed_data functions preprocess data for model training.

```
from t5_utils import preprocess_function, preprocessed_data

train_data, eval_data = preprocessed_data(train_data, eval_data, max_input_length, max_target_length)
```
### 6. Metrics Computation

Functions like **postprocess_text**, **preprocess_logits_for_metrics** are helping function for compute metrics and **compute_metrics** assist in computing metrics during and after training.

```
from t5_utils import compute_metrics

 trainer = transformers.Trainer(
      model=model,
      train_dataset=train_data,
      eval_dataset=eval_data,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics,
      preprocess_logits_for_metrics= preprocess_logits_for_metrics,
```
This is how you use compute metrics in training.

### 7. Training the Model

The train_model function is the main function for training the T5 model on a specific use case.

```
from t5_utils import train_model

train_model(train_data, eval_data, project="Finetunning-t5", base_model_name="T5", model="t5-base", ...)
```
### 8. Inference

The inference function allows for making predictions on new data using a fine-tuned model.

```
from t5_utils import inference

result = inference(task, prompt, fined_tuned_path="/content/T5-Finetunning-t5/checkpoint-100")
```

### 9. Computing Metrics for Testing

If you want to compute metrics separately, use the computing_metrics_for_test function.

```
from t5_utils import computing_metrics_for_test

test_metrics = computing_metrics_for_test(task, model, metrics, test_data)
```

