This repository contains the code for the paper ["PreAdapter: Pre-training Language Models on Knowledge Graphs"](https://link.springer.com/chapter/10.1007/978-3-031-77850-6_12) at ISWC 2024

# Dependencies

Experiments were run using python 3.8

Pip dependencies are listed in the requirements.txt file.

# Running the code
The code can be run using the run.py script.
Run arguments may be adjusted in the config.py file.

The following arguments are used for the specific models run in the paper:

RoBERTa: `--backbone roberta  --baseline one_qa_dragon`

Adapter: `--backbone roberta_adapter  --baseline one_qa_dragon`

K-Adapter: `--backbone roberta_adapter  --baseline one_qa_dragon`

CapsKG: `--backbone roberta_adapter  --baseline ctr_kg_qa_dragon`

PreAdapter: `--backbone roberta_adapter  --baseline ctr_kg_qa_dragon --build_adapter_mask`

# Data
The data used in the experiments are contained in the dat folder.

## Templates
Used templates are contained within the specific dataset folders in templates.json.


## KG
Available Knowledge Graphs (KGs) are:
- ConceptNet CSQA
- ConceptNet OBQA

## QA
Available Question Answering (QA) datasets are:
- CSQA
- OBQA

# Example usage:

`python3 run.py --baseline ctr_kg_qa_dragon --backbone roberta_adapter --task qa_dragon --use_predefine_args --num_train_epochs 5 --max_seq_length=64 --build_adapter_mask --yaml_param_num 0 --output_dir "\my_output_directory" --auto_resume --reuse_data`

# Citation:

If you use this code in your experiments, please cite the following paper:
```bibtex
@inproceedings{omeliyanenko2024preadapter,
  title={PreAdapter: Pre-training Language Models on Knowledge Graphs},
  author={Omeliyanenko, Janna and Hotho, Andreas and Schl{\"o}r, Daniel},
  booktitle={International Semantic Web Conference},
  pages={210--226},
  year={2024},
  organization={Springer}
}
```