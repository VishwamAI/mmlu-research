# Detailed Report on LLaMA: Open and Efficient Foundation Language Models

## Paper Summary
LLaMA is a collection of foundation language models ranging from 7B to 65B parameters. These models are trained on trillions of tokens using publicly available datasets. The LLaMA-13B model outperforms GPT-3 on most benchmarks, and the LLaMA-65B model is competitive with Chinchilla-70B and PaLM-540B. All models are released to the research community.

## Links
- [PDF](https://arxiv.org/pdf/2302.13971v1.pdf)
- [Abstract](https://arxiv.org/abs/2302.13971v1)
- [arXiv 2023 PDF](https://research.facebook.com/file/1574548786327032/LLaMA--Open-and-Efficient-Foundation-Language-Models.pdf)
- [arXiv 2023 Abstract](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/)

## Code Repositories
- [facebookresearch/llama](https://github.com/facebookresearch/llama)
- [huggingface/transformers](https://github.com/huggingface/transformers)
- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)

## Tasks
- Arithmetic Reasoning
- Code Generation
- Common Sense Reasoning
- Few-Shot Learning
- Math Word Problem Solving
- Multi-task Language Understanding
- Question Answering
- Sentence Completion
- Stereotypical Bias Analysis
- Zero-Shot Learning

## Datasets
- Natural Questions
- MMLU
- GSM8K
- C4
- TriviaQA
- HumanEval
- HellaSwag
- BoolQ
- MATH
- PIQA
- WinoGrande
- OpenBookQA
- RACE
- TruthfulQA
- MBPP
- Billion Word Benchmark
- CrowS-Pairs
- ARC
- SIQA
- CCNet
- MedConceptsQA

## Results
- **Common Sense Reasoning**: LLaMA 65B (zero-shot) achieved an accuracy of 56.0 on the ARC (Challenge) dataset, ranking #26.
- **Multi-task Language Understanding**: LLaMA 33B (5-shot) achieved an average accuracy of 57.8%, ranking #54. LLaMA 65B (fine-tuned) achieved an average accuracy of 68.9%, ranking #36. LLaMA 65B (5-shot) achieved an average accuracy of 63.4%, ranking #45.
- **Question Answering**: LLaMA 65B (few-shot, k=5) achieved state-of-the-art performance on Natural Questions and TriviaQA benchmarks.

## Methods
- **Adam Optimizer**: Used for training with specific hyper-parameters.
- **Attention Dropout**: Applied to improve model performance.
- **BPE**: Byte-pair encoding used for tokenization.
- **Cosine Annealing**: Learning rate schedule used during training.
- **Dense Connections**: Implemented to enhance model architecture.
- **Dropout**: Applied to prevent overfitting.
- **Fixed Factorized Attention**: Used to improve attention mechanisms.
- **GELU**: Activation function used in the model.
- **GPT-3**: Comparison model for performance evaluation.
- **Layer Normalization**: Applied to stabilize training.
- **Linear Layer**: Used in the model architecture.
- **Linear Warmup With Cosine Annealing**: Learning rate schedule used during training.
- **LLaMA**: The main model discussed in the paper.
- **Multi-Head Attention**: Used in the transformer architecture.
- **Residual Connection**: Implemented to improve model performance.
- **Scaled Dot-Product Attention**: Used in the attention mechanism.
- **Softmax**: Applied in the attention mechanism.
- **Strided Attention**: Used to improve attention mechanisms.
- **Weight Decay**: Applied to prevent overfitting.

## Detailed Analysis

### Introduction
Large Languages Models (LLMs) trained on massive corpora of texts have shown their ability to perform new tasks from textual instructions or from a few examples. These few-shot properties first appeared when scaling models to a sufficient size, resulting in a line of work that focuses on further scaling these models. These efforts are based on the assumption that more parameters will lead to better performance. However, recent work shows that, for a given compute budget, the best performances are not achieved by the largest models, but by smaller models trained on more data. The objective of the scaling laws is to determine how to best scale the dataset and model sizes for a particular training compute budget. However, this objective disregards the inference budget, which becomes critical when serving a language model at scale. In this context, given a target level of performance, the preferred model is not the fastest to train but the fastest at inference, and although it may be cheaper to train a large model to reach a certain level of performance, a smaller one trained longer will ultimately be cheaper at inference. The focus of this work is to train a series of language models that achieve the best possible performance at various inference budgets, by training on more tokens than what is typically used. The resulting models, called LLaMA, range from 7B to 65B parameters with competitive performance compared to the best existing LLMs. For instance, LLaMA-13B outperforms GPT-3 on most benchmarks, despite being 10× smaller. We believe that this model will help democratize the access and study of LLMs, since it can be run on a single GPU. At the higher-end of the scale, our 65B-parameter model is also competitive with the best large language models such as Chinchilla or PaLM-540B. Unlike Chinchilla, PaLM, or GPT-3, we only use publicly available data, making our work compatible with open-sourcing, while most existing models rely on data which is either not publicly available or undocumented. There exist some exceptions, notably OPT, GPT-NeoX, BLOOM, and GLM, but none that are competitive with PaLM-62B or Chinchilla. In the rest of this paper, we present an overview of the modifications we made to the transformer architecture, as well as our training method. We then report the performance of our models and compare with others LLMs on a set of standard benchmarks. Finally, we expose some of the biases and toxicity encoded in our models, using some of the most recent benchmarks from the responsible AI community.

### Approach
Our training approach is similar to the methods described in previous work, and is inspired by the Chinchilla scaling laws. We train large transformers on a large quantity of textual data using a standard optimizer.

#### Pre-training Data
Our training dataset is a mixture of several sources that cover a diverse set of domains. For the most part, we reuse data sources that have been leveraged to train other LLMs, with the restriction of only using data that is publicly available, and compatible with open sourcing. This leads to the following mixture of data and the percentage they represent in the training set:
- **English CommonCrawl [67%]**: We preprocess five CommonCrawl dumps, ranging from 2017 to 2020, with the CCNet pipeline. This process deduplicates the data at the line level, performs language identification with a fastText linear classifier to remove non-English pages and filters low quality content with an n-gram language model. In addition, we trained a linear model to classify pages used as references in Wikipedia vs. randomly sampled pages, and discarded pages not classified as references.
- **C4 [15%]**: During exploratory experiments, we observed that using diverse pre-processed CommonCrawl datasets improves performance. We thus included the publicly available C4 dataset in our data. The preprocessing of C4 also contains deduplication and language identification steps: the main difference with CCNet is the quality filtering, which mostly relies on heuristics such as presence of punctuation marks or the number of words and sentences in a webpage.
- **Github [4.5%]**: We use the public GitHub dataset available on Google BigQuery. We only kept projects that are distributed under the Apache, BSD and MIT licenses. Additionally, we filtered low quality files with heuristics based on the line length or proportion of alphanumeric characters, and removed boilerplate, such as headers, with regular expressions. Finally, we deduplicate the resulting dataset at the file level, with exact matches.
- **Wikipedia [4.5%]**: We add Wikipedia dumps from the June-August 2022 period, covering 20 languages.
- **Books [4.5%]**: We use the Books3 dataset, which contains a large collection of books.
- **ArXiv [2.5%]**: We include a subset of the ArXiv dataset, focusing on papers from the computer science and mathematics categories.
- **StackExchange [2.0%]**: We use a subset of the StackExchange dataset, focusing on high-quality questions and answers.

### Architecture
Following recent work on large language models, our network is based on the transformer architecture. We leverage various improvements that were subsequently proposed, and used in different models such as PaLM. Here are the main differences with the original architecture, and where we found the inspiration for this change:
- **Pre-normalization [GPT3]**: To improve the training stability, we normalize the input of each transformer sub-layer, instead of normalizing the output. We use the RMSNorm normalizing function.
- **SwiGLU activation function [PaLM]**: We replace the ReLU non-linearity by the SwiGLU activation function to improve the performance. We use a dimension of 2/3 4d instead of 4d as in PaLM.
- **Rotary Embeddings [GPTNeo]**: We remove the absolute positional embeddings, and instead, add rotary positional embeddings (RoPE) at each layer of the network.

### Main Results
Following previous work, we consider zero-shot and few-shot tasks, and report results on a total of 20 benchmarks. We evaluate LLaMA on free-form generation tasks and multiple choice tasks. In the multiple choice tasks, the objective is to select the most appropriate completion among a set of given options, based on a provided context. We select the completion with the highest likelihood given the provided context.

### Common Sense Reasoning
We consider eight standard common sense reasoning benchmarks: BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC easy and challenge, and OpenBookQA. These datasets include Cloze and Winograd style tasks, as well as multiple choice question answering. We evaluate in the zero-shot setting as done in the language modeling community. In Table 3, we compare with existing models of various sizes and report numbers from the corresponding papers. First, LLaMA-65B outperforms Chinchilla-70B on all reported benchmarks but BoolQ. Similarly, this model surpasses PaLM-540B everywhere but on BoolQ and WinoGrande. LLaMA-13B model also outperforms GPT-3 on most benchmarks despite being 10× smaller.

### Massive Multitask Language Understanding
The massive multitask language understanding benchmark, or MMLU, introduced by Hendrycks et al. (2020) consists of multiple choice questions covering various domains of knowledge, including humanities, STEM and social sciences. We evaluate our models in the 5-shot setting, using the examples provided by the benchmark, and report results in Table 9. On this benchmark, we observe that the LLaMA-65B is behind both Chinchilla-70B and PaLM-540B by a few percent in average, and across most domains. A potential explanation is that we have used a limited amount of books and academic papers in our pre-training data, i.e., ArXiv, Gutenberg and Books3, that sums up to only 177GB, while these models were trained on up to 2TB of books. This large quantity of books used by Gopher, Chinchilla and PaLM may also explain why Gopher outperforms GPT-3 on this benchmark, while it is comparable on other benchmarks.

### Bias, Toxicity and Misinformation
Large language models have been shown to reproduce and amplify biases that are existing in the training data, and to generate toxic or offensive content. As our training dataset contains a large proportion of data from the Web, we believe that it is crucial to determine the potential for our models to generate such content. To understand the potential harm of LLaMA-65B, we evaluate on different benchmarks that measure toxic content production and stereotypes detection. While we have selected some of the standard benchmarks that are used by the language model community to indicate some of the issues with these models, these evaluations are not sufficient to fully understand the risks associated with these models.

### Carbon Footprint
The training of our models has consumed a massive quantity of energy, responsible for the emission of carbon dioxide. We follow the recent literature on the subject and breakdown both the total energy consumption and the resulting carbon footprint. We follow a formula to estimate the Watt-hour, Wh, needed to train a model, as well as the tons of carbon emissions, tCO2 eq. For the Wh, we use the formula: Wh = GPU-h×(GPU power consumption)×PUE, where we set the Power Usage Effectiveness (PUE) at 1.1. The resulting carbon emission depends on the location of the data center used to train the network. For instance, BLOOM uses a grid that emits 0.057 kg CO2 eq/KWh leading to 27 tCO2 eq and OPT a grid that emits 0.231 kg CO2 eq/KWh, leading to 82 tCO2 eq. In this study, we are interested in comparing the cost in carbon emission of training these models if they were trained in the same data center. Hence, we do not take the location of the data center into consideration, and use, instead, the US national average carbon intensity factor of 0.385 kg CO2 eq/KWh. This leads to the following formula for the tons of carbon emissions: tCO2 eq = MWh × 0.385.

## Conclusion
The LLaMA models are competitive with state-of-the-art foundation models while being trained exclusively on publicly available data. The authors hope that releasing these models will accelerate the development of large language models and help improve their robustness and mitigate issues like toxicity and bias. Future work includes finetuning these models on instructions and releasing larger models trained on larger pretraining corpora.
