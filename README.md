# Deep Spell Checker

DeepSp is a Deep learning based model for spell checking perisna texts

## packages Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install packages.

```bash
pip install -r requirements.txt
```
## Download model dump


```bash
pip install gdown
gdown https://drive.google.com/uc?id=1VNr27NoyJ0H8Vt3-ck2r9UF4PrKDb1nB
cp bert_spell_checker_checkpoint.tar.xz model/
tar xvf model/bert_spell_checker_checkpoint.tar.xz
```
## run werbservice


```bash
python webservice.py
```

