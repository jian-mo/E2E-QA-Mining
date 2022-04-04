# Question Answer Pair Mining  using 🤗transformers

## Project Overview

This project is aim to train an end to end model for extracting question and answer pairs from a given passage. 
Many other researches use a pipeline approach like [patil-suraj](https://github.com/patil-suraj/question_generation). Inspired by the [bigsciece](https://github.com/~~bigscience-workshop/bigscience),
our project proves that transformer models can also solving the qa-mining problem in a direct end to end fashion by providing the model 
with a task prompt. We followed the training scheme of [patil-suraj](https://github.com/patil-suraj/question_generation)
and adapted the training script to be e2e. For more usage of their training script please visit the aforementioned link.

## Requirements
```
transformers==3.0.0
nltk
nlp==0.2.0 # only if you want to fine-tune.
```

after installing `nltk` do
```bash
python -m nltk.downloader punkt
```


#### Question Answer Pair Generation



```
 generator = pipeline('text2text-generation', model='mojians/E2E-QA-Mining', device=local_rank)

 generator.model = deepspeed.init_inference(generator.model,
                                            mp_size=world_size,
                                            dtype=torch.float,
                                            replace_method='auto')
 string = generator("context: The English name 'Normans' comes from the French words Normans/Normanz, plural of Normant, modern French normand, which is itself borrowed from Old Low Franconian Nortmann 'Northman' or directly from Old Norse Norðmaðr, Latinized variously as Nortmannus, Normannus, or Nordmannus (recorded in Medieval Latin, 9th century) to mean 'Norseman, Viking'. generate questions and answers:", do_sample=True, min_length=50,max_length=300)

```





## Fine-tuning

### Data processing 

To support different data formats the trainer expects pre-processed cached dataset, so you can process the data the way you want.
The cached dataset should be saved using `torch.save` and it should return a `dict` with `source_ids`, `target_ids`, `attention_mask` keys from `__getitem__`.

- `source_ids`: encoded source text
- `target_ids`: encoded target text
- `attention_mask`: attention mask for the `source_ids`
  
The `T2TDataCollator` takes care of preparing right `input_ids` and `labels`. It also trims the batches dynamically to remove excessive padding tokens, to speed up the training.

The `data/squad_multitask` containes the modifed SQuAD dataset for answer aware question generation (using both prepend and highlight formats), question answering (text-to-text), answer extraction and end-to-end question generation. This dataset can be loaded using the awesome 🤗`nlp` library, this makes processing very easy.

To process and cache the dataset use `prepare_data.py` script. It will load the correct tokenizer depending on the `model_type` argument. It adds two new tokens `<sep>` and `<hl>` to the tokenizer and saves it at `{model_type}_qg_tokenizer` path. You should pass this tokenizer to the fine-tuning script.

The datasets will be saved in `data/` directory. You should provide filenames using `train_file_name` and `valid_file_name` arguments.


```

**process data for end2end question answer mining with highlight_qg_format**

`valid_for_qg_only` argument is used to decide if the validation set should only contain data for qg task. For my multi-task experiments I used validation data with only qg task so that the eval loss curve can be easly compared with other single task models

```bash
python prepare_data.py \
    --task aeqg \
    --valid_for_qg_only \ 
    --model_type t5 \
    --dataset_path data/squad_multitask/ \
    --qg_format highlight_qg_format \
    --max_source_length 512 \
    --max_target_length 32 \
    --train_file_name train_data_qa_qg_hl_t5.pt \
    --valid_file_name valid_data_qg_hl_t5.pt \
```


### training
Use the `run_qg.py` script to  start training. It uses transformers `Trainer` class for training the models.


```bash
python run_qg.py \
    --model_name_or_path t5-small \
    --model_type t5 \
    --tokenizer_name_or_path data/squad_multitask/t5_qg_tokenizer \
    --output_dir t5-small-qg-hl \
    --train_file_path data/train_data_qg_hl_t5.pt \
    --valid_file_path data/valid_data_qg_hl_t5.pt \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --seed 42 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --logging_steps 100
```

or if you want to train it from script or notebook then

```python3
from run_qg import run_qg

args_dict = {
    "model_name_or_path": "t5-small",
    "model_type": "t5",
    "tokenizer_name_or_path": "t5_qg_tokenizer",
    "output_dir": "t5-small-qg-hl",
    "train_file_path": "data/train_data_qg_hl_t5.pt",
    "valid_file_path": "data/valid_data_qg_hl_t5.pt",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "num_train_epochs": 10,
    "seed": 42,
    "do_train": True,
    "do_eval": True,
    "evaluate_during_training": True,
    "logging_steps": 100
}

# start training
run_qg(args_dict)
```

## Demo

Try out our model in this huggingface space [https://huggingface.co/spaces/mojians/E2E-QA-mining](https://huggingface.co/spaces/mojians/E2E-QA-mining)
