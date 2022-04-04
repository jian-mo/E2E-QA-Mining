import os
import deepspeed
import torch
from transformers import pipeline

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
generator = pipeline('text2text-generation', model='t5-small-aeqg-hl', device=local_rank)


#
# generator.model = deepspeed.init_inference(generator.model,
#                                            mp_size=world_size,
#                                            dtype=torch.float,
#                                            replace_method='auto')

string = generator("context: The English name 'Normans' comes from the French words Normans/Normanz, plural of Normant, modern French normand, which is itself borrowed from Old Low Franconian Nortmann 'Northman' or directly from Old Norse Norðmaðr, Latinized variously as Nortmannus, Normannus, or Nordmannus (recorded in Medieval Latin, 9th century) to mean 'Norseman, Viking'. generate questions and answers:", do_sample=True, min_length=50,max_length=300)
print(string)

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    while True:
        input_str=input("input:")
        context_str="context: "+ input_str + "  generate questions and answers: "
        print(context_str)
        string_input= generator(context_str, do_sample=True, min_length=50,max_length=100)
        print(string_input)
