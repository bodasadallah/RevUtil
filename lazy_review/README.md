# Lazy Review Inference

This folder contains scripts for running inference on the lazy thinking classification task using fine-tuned models.

## Quick Start

### Running Inference

1. Make the bash script executable:
```bash
chmod +x run_inference.sh
```

2. Run the inference:
```bash
./run_inference.sh
```

The script will:
- Load the model from Hugging Face: `sukannya/lazy_review_plus_emnlp_Qwen2.5-7B-Instruct`
- Process the test data from `data/lazy_thinking_fine_grained_test_with_eg.jsonl`
- Save results to `outputs/` directory

### Output

The results will be saved as a TSV file with the following columns:
- `id`: Dataset identifier
- `title`: Sample identifier
- `review`: The review text (without newlines)
- `weakness`: The target sentence to classify
- `mapping`: Ground truth label
- `outputs`: Model prediction
- `model`: Model path used for inference

## Input Data Structure

### JSONL Format

The input data should be in JSONL format (one JSON object per line). Each line represents one sample with the following structure:

```json
{
  "dataset": "lazy_reviewing_fine_grained_data",
  "id": "lazy_thinking_73",
  "messages": [
    {
      "role": "user",
      "content": "FULL_PROMPT_HERE"
    },
    {
      "role": "assistant",
      "content": "LABEL_HERE"
    }
  ]
}
```

### Fields Explanation

1. **dataset** (string): A dataset identifier (e.g., "lazy_reviewing_fine_grained_data")

2. **id** (string): A unique identifier for the sample (e.g., "lazy_thinking_73")

3. **messages** (array): An array containing two message objects:
   - **User message**: Contains the full prompt
   - **Assistant message**: Contains the ground truth label

### Prompt Structure

The user message content follows this structure:

```
[INSTRUCTION WITH LAZY THINKING DEFINITIONS]

Given a target sentence, you need to classify the target sentence into one of the lazy thinking classes 
eg., 'The topic is too niche', 'The results are negative'.

Target Sentence: [SENTENCE_TO_CLASSIFY]
```

The instruction includes:
- Background about lazy thinking in peer reviewing
- A dictionary of lazy thinking classes with their meanings and reasons for being problematic
- Task description
- The target sentence to classify

### Lazy Thinking Classes

The model classifies review sentences into these categories:
- **The results are not surprising**
- **The results contradict what I would expect**
- **The results are not novel**
- **This has no precedent in existing literature**
- **The results do not surpass the latest SOTA**
- **The results are negative**
- **This method is too simple**
- **The paper doesn't use [my preferred methodology], e.g., deep learning**
- **The topic is too niche/ Narrow Topics**
- **The approach is tested only on [not English], so unclear if it will generalize to other languages**
- **The paper has language errors/ Writing style**
- **The paper is missing the comparison to the [latest X]**
- **The authors could also do [extra experiment X]**
- **The authors should have done [X] instead**
- **None** (not lazy thinking)
- **Not Enough Info**
- **Non Mainstream approaches**
- **Resource paper**

## Creating Your Own Test Samples

To evaluate your own samples, create a JSONL file following this template:

```json
{
  "dataset": "your_dataset_name",
  "id": "sample_1",
  "messages": [
    {
      "role": "user",
      "content": " As the number of scientific publications grows over the years, the need for efficient quality control in peer-reviewing \nbecomes crucial. Due to the high reviewing load, the reviewers are often prone to different kinds of bias, which inherently contribute to \nlower reviewing quality and this overall hampers the scientific progress in the field. The ACL peer reviewing guidelines characterize some \nof such reviewing bias.  These are termed lazy thinking. The lazy thinking classes, their meaning and the reason for them being problematic \nare provided as a dictionary in the format ```key: meaning: reason for being problematic```. If a meaning or reason for being problematic is not known,\n'' is provided in that place: \n\n\n{'The results are not surprising': 'Many findings seem obvious in retrospect, but this does not mean that the community is already aware of them and can use them as building blocks for future work.',\n'The results contradict what I would expect': 'You may be a victim of confirmation bias, and be unwilling to accept data contradicting your prior beliefs.',\n'The results are not novel': 'Such broad claims need to be backed up with references.',\n'This has no precedent in existing literature': 'Believe it or not: papers that are more novel tend to be harder to publish. Reviewers may be unnecessarily conservative.',\n'The results do not surpass the latest SOTA', 'SOTA results are neither necessary nor sufficient for a scientific contribution. An engineering paper could also offer improvements on other dimensions (efficiency, generalizability, interpretability, fairness, etc.',\n'The results are negative': 'The bias towards publishing only positive results is a known problem in many fields, and contributes to hype and overclaiming. If something systematically does not work where it could be expected to, the community does need to know about it.',\n'This method is too simple': 'The goal is to solve the problem, not to solve it in a complex way. Simpler solutions are in fact preferable, as they are less brittle and easier to deploy in real-world settings.',\n'The paper doesn't use [my preferred methodology], e.g., deep learning': 'NLP is an interdisciplinary field, relying on many kinds of contributions: models, resource, survey, data\/linguistic\/social analysis, position, and theory.',\n'The topic is too niche\/ Narrow Topics': 'It is easier to publish on trendy,\u2018scientifically sexy\u2019 topics (Smith, 2010). In the last two years, there has been little talk of anything other than large pretrained Transformers,\nwith BERT alone becoming the target of over 150 studies proposing analysis and various modifications (Rogers et al., 2020). The \u2018hot trend\u2019 forms the prototype for the kind of paper that should be\nrecommended for acceptance. Niche topics such as historical text normalization are downvoted (unless, of course, BERT could somehow be used for that).',\n'The approach is tested only on [not English], so unclear if it will generalize to other languages': 'The same is true of NLP research that tests only on English. Monolingual work on any language is important both practically (methods and resources for that language) and theoretically (potentially contributing to deeper understanding of language in general)',\n'The paper has language errors\/ Writing style': 'As long as the writing is clear enough, better scientific content should be more valuable than better journalistic skills. If it comes in the comments, suggestions and typos, not lazy. The paper\u2019s language or writing style. Please focus on the paper\u2019s substance. We understand that there may be times when the language or writing style is so poor that reviewers can not understand the paper\u2019s content and substance. In that case, it is fine to reject the paper, however you should only do so after making a concerted effort to understand the paper.',\n'The paper is missing the comparison to the [latest X]': 'Per ACL policy, the authors are not obliged to draw comparisons with contemporaneous work, i.e., work published within three months before the submission (or three months before a re-submission).',\n'The authors could also do [extra experiment X]': 'It is always possible to come up with extra experiments and follow-up work. This is fair if the experiments that are already presented are insufficient for the claim that the authors are making. But any other extra experiments are in the \u201cnice-to-have\u201d category, and belong in the \u201csuggestions\u201d section rather than \u201cweaknesses.\u201d',\n'The authors should have done [X] instead': 'A.k.a. \u201cI would have written a different paper.\u201d There are often several valid approaches to a problem. This criticism applies only if the authors\u2019 choices prevent them from answering their research question, their framing is misleading, or the question is not worth asking. If not, then [X] is a comment or a suggestion, but not a \u201cweakness.\u201d',\n'None': 'This is not lazy thinking',\n'Not Enough Info': 'Problematic but not sure how to verify from the given review'\n'Non Mainstream approaches' : 'Since a \u2018mainstream\u2019 *ACL paper currently uses DL-based methods, anything else might look like it does not really\nbelong in the main track - even though ACL stands for \u2018Association for Computational Linguistics\u2019. That puts interdisciplinary efforts at a disadvantage,\nand continues the trend for intellectual segregation of NLP (Reiter, 2007). E.g., theoretical papers and linguistic resources should not be a priori at a\ndisadvantage just because they do not contain DL experiments.',\n'Resource paper': 'The paper is a resource paper. In a field that relies on supervised machine learning as much as NLP, development of datasets is as important as modeling work. This blog post discusses what can and cannot be grounds for dismissing a resource paper'\n} \n\n\n\nGiven a target sentence, you need to classify the target sentence into one of the lazy thinking classes \neg., 'The topic is too niche', 'The results are negative'.\n\n\nTarget Sentence: YOUR_REVIEW_SENTENCE_HERE\n\n"
    },
    {
      "role": "assistant",
      "content": "YOUR_GROUND_TRUTH_LABEL_HERE"
    }
  ]
}
```

**Important Notes:**
1. Keep the full instruction text exactly as shown (it's part of the model's training)
2. Only replace `YOUR_REVIEW_SENTENCE_HERE` with your target sentence
3. The assistant content can be any of the lazy thinking classes or "None" if you have ground truth
4. If you don't have ground truth, you can put an empty string or placeholder

### Example: Custom Sample

```json
{
  "dataset": "my_custom_dataset",
  "id": "custom_1",
  "messages": [
    {
      "role": "user",
      "content": "[FULL_INSTRUCTION_AS_ABOVE]\n\nTarget Sentence: The paper uses outdated baselines from 2019 and doesn't compare with recent SOTA models.\n\n"
    },
    {
      "role": "assistant",
      "content": "The paper is missing the comparison to the [latest X]"
    }
  ]
}
```

## Custom Inference Command

To run inference with your own data file:

```bash
python eval.py \
    --dataset path/to/your/data.jsonl \
    --model_path sukannya/lazy_review_plus_emnlp_Qwen2.5-7B-Instruct \
    --output_dir outputs/
```

Or modify the `run_inference.sh` script to point to your custom data file.

## Requirements

- Python 3.8+
- vllm
- transformers
- datasets
- torch
- pandas
- python-dotenv (optional, for HuggingFace token)

Install dependencies:
```bash
pip install vllm transformers datasets torch pandas python-dotenv
```

## Notes

- The model uses vLLM for efficient inference
- Generation parameters: temperature=0, max_tokens=100
- The model runs on GPU (requires CUDA)
- For LoRA merged models, add the `--merged_lora` flag

## Troubleshooting

**Issue**: Model not found
- **Solution**: Ensure you have access to the Hugging Face model or that it's publicly available

**Issue**: Out of memory
- **Solution**: Reduce batch size or use a smaller model

**Issue**: CUDA errors
- **Solution**: Ensure you have a compatible GPU and CUDA installation
