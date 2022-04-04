import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, HfArgumentParser, TrainingArguments
import deepspeed

from data_collator import T2TDataCollator

logger = logging.getLogger(__name__)


@dataclass
class EvalArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # local_rank: Optional[int] = field(
    #     metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    # )

    valid_file_path: str = field(
        metadata={"help": "Path for cached valid dataset"}
    )
    model_type: str = field(metadata={"help": "One of 't5', 'bart'"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    num_beams: Optional[int] = field(
        default=6,
        metadata={"help": "num_beams to use for decoding"}
    )
    max_decoding_length: Optional[int] = field(
        default=500,
        metadata={"help": "maximum length for decoding"}
    )
    # output_dir: Optional[str] = field(
    #     default="hypothesis.txt",
    #     metadata={"help": "path to save the generated questions."}
    # )


    rundeepspeed: Optional[bool] = field(
        default=True
    )


    do_sample: Optional[bool] = field(
        default=True
    )





def get_predictions(model, tokenizer, data_loader, args,num_beams=4, max_length=32, length_penalty=1,device='cuda',do_sample=True):
    model.to(device)
    
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            if not do_sample:
                if args.rundeepspeed:
                    outs = model.generate(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        num_beams=num_beams,
                        max_length=max_length,
                        length_penalty=length_penalty,
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=6,

                    )
                else:
                    outs = model.generate(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        num_beams=num_beams,
                        max_length=max_length,
                        length_penalty=length_penalty,
                        repetition_penalty=1,
                        no_repeat_ngram_size=6,

                    )
            else:
                if args.rundeepspeed:
                    outs = model.generate(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        do_sample=True,
                        max_length=max_length,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=3,
                        repetition_penalty=1,
                        no_repeat_ngram_size=6,

                    )
                else:
                    outs = model.generate(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        do_sample=True,
                        max_length=max_length,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=3,
                        repetition_penalty=1,
                        no_repeat_ngram_size=6,
                    )

            prediction = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            print(prediction)
            predictions.extend(prediction)

    return predictions

def main():
    parser = HfArgumentParser((EvalArguments,TrainingArguments))
    eval_args,train_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        eval_args.tokenizer_name_or_path if eval_args.tokenizer_name_or_path else eval_args.model_name_or_path,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(eval_args.model_name_or_path)

    valid_dataset = torch.load(eval_args.valid_file_path)
    collator = T2TDataCollator(
        tokenizer=tokenizer,
        model_type=eval_args.model_type,
        mode="inference"
    )
    #
    # local_rank = int(os.getenv('LOCAL_RANK', '0'))
    # world_size = int(os.getenv('WORLD_SIZE', '1'))


    if eval_args.rundeepspeed:
        device=train_args.local_rank
    else:
        device = 'cuda' if torch.cuda.is_available else 'cpu'



    loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, collate_fn=collator)

    if eval_args.rundeepspeed:
        ds_engine = deepspeed.init_inference(model,
                                             mp_size=1,
                                             dtype=torch.half,
                                             checkpoint=None ,
                                             replace_method='auto')
        model = ds_engine.module


    predictions = get_predictions(
        model=model,
        tokenizer=tokenizer,
        data_loader=loader,
        args=eval_args,
        num_beams=eval_args.num_beams,
        max_length=eval_args.max_decoding_length,
        device=device,
        do_sample=eval_args.do_sample,
    )

    with open(train_args.output_dir, 'w',encoding='utf-8') as f:
        f.write("\n".join(predictions))
    
    logging.info(f"Output saved at {train_args.output_dir}")


if __name__ == "__main__":
    main()
