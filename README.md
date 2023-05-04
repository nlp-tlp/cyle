# CylE: Cylinder Embeddings for Multi-hop Reasoning over Knowledge Graphs

This is the codebase for the paper [CylE: Cylinder Embeddings for Multi-hop Reasoning over Knowledge Graphs](https://aclanthology.org/2023.eacl-main.127/) (EACL 2023).

## Getting started

### Step 1: Data Preparation

- Download the datasets [here](http://snap.stanford.edu/betae/KG_data.zip), then move `KG_data.zip` to `./cyle/` directory

- Unzip `KG_data.zip` to `./cyle/data/`:
  ```
  cd cyle/
  unzip -d data KG_data.zip
  ```

### Step 2: Installing Requirements

- If you are familiar with `pip` or `conda`, please install requirements by your own preference:

  ```
  python=3.8
  pytorch=1.9.1
  tqdm
  tensorboardX
  ```

- [Optional] For those prefer to use `Anaconda`, you can follow these:
  ```
  conda create --name cyle python=3.8.11
  conda activate cyle
  conda install pytorch==1.9.1 cudatoolkit=10.2 -c pytorch
  conda install tqdm
  pip install tensorboardX
  ```

<!-- conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch -->

### Step 3: Training Model

- Run script `./run.sh`

- Type `[1-3]` to select which dataset to run model, type `[4/q/quit/exit]` to exit

## Citation

If you find this code useful for your research, please consider citing the following paper:

```
@inproceedings{nguyen2023cyle,
    title = "{C}yl{E}: Cylinder Embeddings for Multi-hop Reasoning over Knowledge Graphs",
    author = "Nguyen, Chau Duc Minh  and
      French, Tim  and
      Liu, Wei  and
      Stewart, Michael",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.127",
    pages = "1736--1751",
}
```

## Acknowledgement

We acknowledge the code of [KGReasoning](https://github.com/snap-stanford/KGReasoning) for their contributions.
