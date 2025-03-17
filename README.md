# Analyzing Memorization in Large Language Models through the Lens of Model Attribution
[![arXiv](https://img.shields.io/badge/arXiv-2501.05078-b31b1b.svg)](https://arxiv.org/abs/2501.05078)

Code for our NAACL paper
## Abstract
Large Language Models (LLMs) are prevalent in modern applications but often memorize training data, leading to privacy breaches and copyright issues. Existing research has mainly focused on post-hoc analyses—such as extracting memorized content or developing memorization metrics—without exploring the underlying architectural factors that contribute to memorization. In this work, we investigate memorization from an architectural lens by analyzing how attention modules at different layers impact its memorization and generalization performance. Using attribution techniques, we systematically intervene in the LLM's architecture by bypassing attention modules at specific blocks while keeping other components like layer normalization and MLP transformations intact. We provide theorems analyzing our intervention mechanism from a mathematical view, bounding the difference in layer outputs with and without our attributions. Our theoretical and empirical analyses reveal that attention modules in deeper transformer blocks are primarily responsible for memorization, whereas earlier blocks are crucial for the model's generalization and reasoning capabilities. We validate our findings through comprehensive experiments on different LLM families (Pythia and GPT-Neo) and five benchmark datasets. Our insights offer a practical approach to mitigate memorization in LLMs while preserving their performance, contributing to safer and more ethical deployment in real-world applications. 

## Installation
Download the repo and install using poetry
```bash
git clone https://github.com/AikyamLab/llm-memorization
cd llm-memorization
poetry install
```
or pip
```bash
git clone https://github.com/AikyamLab/llm-memorization
cd llm-memorization
pip3 install -e .
```

## Running Experiments
Our experiments consist of two steps - 
- Computing memorization scores when short circuiting each attention block of the LLM
- Computing benchmark scores when short-circuiting each attention block of the LLM

### Computing Memorization Scores
Run the following command. The data must be a 2D numpy array where each row is a memorized sample, already tokenized using the model's corresponding tokenizer (See [this file](memorized_data/EleutherAI_gpt-neo-1.3B.npy) for an example). The prefix length can be set as shown below. 
```python
python evaluate_memorization.py  --model-path PATH_TO_HUGGINGFACE_MODEL \
                              --num-layers NUMBER_OF_TRANSFORMER_LAYERS \
                              --prefix-len PREFIX_LEN
                              --batch-size BATCH_SIZE
```

### Computing Benchmark Scores
Run the following command. Refer to [LM Evaluation Harness]() for the list of available tasks. Only two model families are currently supported - GPTNeo and Pythia. Please raise an issue or open a pull request for adding new models architecture families!
```python
python evaluate_benchmarks.py --tasks TASK_LIST \
                              --model-path PATH_TO_HUGGINGFACE_MODEL \
                              --num-layers NUMBER_OF_TRANSFORMER_LAYERS
```
> Please refer to the individual scripts for a full list of options

## Contributing
We welcome contributions to enhance our work! If you have suggestions or find issues, please submit a pull request or open an issue in this repository.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
This work would not be possible without the awesome [GPTNeo](https://github.com/EleutherAI/gpt-neo) and [Pythia](https://github.com/EleutherAI/pythia) models, from [EleutherAI](https://www.eleuther.ai/). The fully open-source nature of these models, including the availability of training data is crucial in enabling the study of memorization of training data in our work. 
All our benchamrking was run using [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) (from EleutherAI again!), which enabled us to have a stable evaluation environment.

## Citation
```
@article{menta2025analyzingmemorizationlargelanguage,
      title={Analyzing Memorization in Large Language Models through the Lens of Model Attribution}, 
      author={Tarun Ram Menta and Susmit Agrawal and Chirag Agarwal},
      year={2025},
      eprint={2501.05078},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.05078}, 
}
```
