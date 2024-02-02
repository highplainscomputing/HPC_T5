
# Fine-Tuning T5 Model for Sentiment Analysis using Transformers Library

## Table of Contents

- [Overview](#Overview)
- [Prerequisites and Environment](#Prerequisites-and-Environment)
- [Data](#Data)
- [Summarization Model](#Summarization-Model)
- [Visualization using WandB](#Visualization-using-WandB)
- [Fine-tunning ](#Fine-tunning )
- [Step-wise Training Trends](#Step-wise-Training-Trends)
- [Evaluation](#Evaluation ) 
- [System Metrics](#System-Metrics)
- [Inference with Fine-tuned Model](#Inference-with-Fine-tuned-Model)
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
We used the [imdb](https://huggingface.co/datasets/imdb) Dataset to fine-tune the model. The is an  Extensive Movie Review Dataset that is designed for binary sentiment classification and surpasses the scale of earlier benchmark datasets. 

                                                    Text: 
                          "I can't believe that those praising this movie herein aren't   
                          thinking of some other film. I was prepared for the possibility 
                          that this would be awful, but the script (or lack thereof) makes 
                          for a film that's also pointless. On the plus side, the general 
                          level of craft on the part of the actors and technical crew is 
                          quite competent, but when you've got a sow's ear to work with you 
                          can't make a silk purse. Ben G fans should stick with just about 
                          any other movie he's been in. Dorothy S fans should stick to 
                          Galaxina. Peter B fans should stick to Last Picture Show and 
                          Target. Fans of cheap laughs at the expense of those who seem to 
                          be asking for it should stick to Peter B's amazingly awful book, 
                          Killing of the Unicorn."
                                                    
                          ..................................................................
                                                    Label:
                                                      0

This dataset comprises a collection of 25,000 strongly polar movie reviews for training purposes and an additional 25,000 for testing. Moreover, there is supplementary unlabeled data available for further applications. The data comprises three splits as shown

                                    DatasetDict({
                                        train: Dataset({
                                            features: ['text', 'label'],
                                            num_rows: 25000
                                        })
                                        test: Dataset({
                                            features: ['text', 'label'],
                                            num_rows: 25000
                                        })
                                        unsupervised: Dataset({
                                            features: ['text', 'label'],
                                            num_rows: 50000
                                        })
                                    })





## Sentiment Analysis Model

[Sentiment analysis](https://huggingface.co/blog/sentiment-analysis-python), a field within natural language processing (NLP), is dedicated to understanding the sentiments conveyed in textual data, categorizing them as positive, negative, or neutral. This powerful technique finds widespread use across various industries due to its ability to extract valuable insights from the opinions and emotions expressed in text. 
![image](https://github.com/highplainscomputing/HPC_T5/assets/150230209/581d1dae-e13e-45c4-a189-f6d3867c917e)


## Visualization using WandB
Understanding the data or the model's behavior is crucial, and visualization plays a key role in achieving this. We have utilized an open-source [WandB(Weights & Biases)](https://wandb.ai/) to gain insight and visualize the model performance.

## Evaluation Metrics
Selecting the appropriate metric is vital for evaluating the performance of any model. In our summarization task, we have employed the [Rouge Score](https://huggingface.co/spaces/evaluate-metric/rouge) to assess the effectiveness of our model.
## Fine-tunning 
![image](https://github.com/highplainscomputing/HPC_T5/assets/150230209/c9d48c0d-e22a-4758-ad26-cf9f507b707c)

## Step-wise Training Trends:
![image](https://github.com/highplainscomputing/HPC_T5/assets/150230209/c82c996e-cc64-4f3b-b75f-076f9a99b3d4)


## Evaluation 
![image](https://github.com/highplainscomputing/HPC_T5/assets/150230209/893afb86-b4d8-4398-b011-b02dbf66126b)


## System Metrics
![image](https://github.com/highplainscomputing/HPC_T5/assets/150230209/e1d2c3f4-29bb-456d-882a-4dc850ac041c)


## Inference with Fine-tuned Model
                                          Dialogue: 
                                          summarize : 
                                          Eric: MACHINE!
                                          Rob: That's so gr8!
                                          Eric: I know! And shows how Americans see Russian ;)
                                          Rob: And it's really funny!
                                          Eric: I know! I especially like the train part!
                                          Rob: Hahaha! No one talks to the machine like that!
                                          Eric: Is this his only stand-up?
                                          Rob: Idk. I'll check.
                                          Eric: Sure.
                                          Rob: Turns out no! There are some of his stand-ups on youtube.
                                          Eric: Gr8! I'll watch them now!
                                          Rob: Me too!
                                          Eric: MACHINE!
                                          Rob: MACHINE!
                                          Eric: TTYL?
                                          Rob: Sure :)
                                          ...........................
                                          Generated Summary:
                                          Eric and Rob are watching a stand-up on youtube.
## Conclusion

In this code, we demonstrate the process of fine-tuning the T5 model for sentiment analysis using the Hugging Face Transformers library. By adhering to the provided steps in the code, you can fine-tune your own T5 model to effectively analyze and categorize sentiments in given textual inputs, providing valuable insights into positive, and negative.

