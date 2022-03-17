# DadmaCheck: A Bert-Based Spell-Checker for Persian


## packages Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install packages.

```bash
pip install -r requirements.txt
```
## Download model checkpoint and vocab


```bash
pip install gdown
gdown https://drive.google.com/uc?id=1VNr27NoyJ0H8Vt3-ck2r9UF4PrKDb1nB
cp bert_spell_checker_checkpoint.tar.xz model/
tar xvf model/bert_spell_checker_checkpoint.tar.xz
```
## run


```bash
python main.py
```

