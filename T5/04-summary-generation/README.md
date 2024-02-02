
# Fine-Tuning T5 Model for Summarization using Transformers Library

## Table of Contents

- [Overview](#Overview)
- [Prerequisites and Environment](#Prerequisites-and-Environment)
- [Data](#Data)
- [Summarization Model](#Summarization-Model)
- [Visualization using WandB](#Visualization-using-WandB)
- [Training Metrics](#Training-Metrics)
- [Evaluation Metrics](#Evaluation-Metrics)
- [System Metrics](#System-Metrics)
- [Evaluation](#Evaluation)
- [Conclusion](#Conclusion)

## Overview
We have fine-tuned the [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) model to excel in summarization tasks. This fine-tuned model can proficiently generate concise and meaningful summaries
## Prerequisites and Environment
Our first step involves installing all the required dependencies.
                                    
                                    !pip install -q accelerate -U
                                    !pip install -q -U datasets
                                    !pip install -q scipy
                                    !pip install -q ipywidgets
                                    !pip install -q wandb
                                    !pip install -q transformers
                                    !pip install -q torch
                                    !pip install -q sentencepiece
                                    !pip install -q tqdm
                                    !pip install -q evaluate
                                    !pip install -q rouge_score
## Data 
We used the [Samsum Dataset](https://huggingface.co/datasets/samsum) to fine-tune the model. The dataset includes chat dialogues, with its summary as shown below. This diverse chat content helps train the model to create effective summaries for conversations.

                                 Dialogue: 
                                 
                                 Olivia: Who are you voting for in this election? 
                                 Oliver: Liberals as always.
                                 Olivia: Me too!!
                                 Oliver: Great)
                                 ............................................
                           
                                 Summary:
                                 
                                 Olivia and Olivier are voting for liberals in this election. 


The dataset consists of three splits as shown below

                              DatasetDict({
                                  train: Dataset({
                                      features: ['id', 'dialogue', 'summary'],
                                      num_rows: 14732
                                  })
                                  test: Dataset({
                                      features: ['id', 'dialogue', 'summary'],
                                      num_rows: 819
                                  })
                                  validation: Dataset({
                                      features: ['id', 'dialogue', 'summary'],
                                      num_rows: 818
                                  })
                              })





## Summarization Model

A brief description of what this project does and who 
In natural language processing (NLP), [Summarization](https://huggingface.co/docs/transformers/tasks/summarization) refers to the process of condensing a longer piece of text into a shorter, coherent version while retaining the most important information. The goal of summarization is to provide a concise representation of the essential content, making it easier for readers to understand the main ideas without going through the entire document.
![image](https://github.com/highplainscomputing/HPC_T5/assets/150230209/9cc84653-6ae2-4c04-b691-5fabc2632907)

## Visualization using WandB
Understanding the data or the model's behavior is crucial, and visualization plays a key role in achieving this. We have utilized an open-source [WandB(Weights & Biases)](https://wandb.ai/) to gain insight and visualize the model performance.

## Evaluation Metrics
Selecting the appropriate metric is vital for evaluating the performance of any model. In our summarization task, we have employed the [Rouge Score](https://huggingface.co/spaces/evaluate-metric/rouge) to assess the effectiveness of our model.
## Fine-tunning 
![image](https://github.com/highplainscomputing/HPC_T5/assets/150230209/d85cc958-bb99-4c9f-94a7-08a1351fbab4)

## Evaluation 
## System Metrics
## Evaluation
## Conclusion



