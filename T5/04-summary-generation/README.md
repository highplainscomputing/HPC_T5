## Overview
We have fine-tuned the [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) model to excel in summarization tasks. This fine-tuned model can proficiently generate concise and meaningful summaries

## Summarization 

A brief description of what this project does and who 
In natural language processing (NLP), summarization refers to the process of condensing a longer piece of text into a shorter, coherent version while retaining the most important information. The goal of summarization is to provide a concise representation of the essential content, making it easier for readers to understand the main ideas without going through the entire document.

## Dataset 
We used the [Samsum Dataset](https://huggingface.co/datasets/samsum) to fine-tune the model. The dataset includes chat dialogues, with its summary as shown below. This diverse chat content helps train the model to create effective summaries for conversations.
   
   ![image](https://github.com/highplainscomputing/HPC_T5/assets/150230209/9ef65ae5-5525-4ea0-aca2-1d80e33f28af)


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





