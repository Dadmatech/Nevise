# Nevise: A Bert-Based Spell-Checker for Persian

Nevise is a Persian spelling-checker developed by Dadmatech  Co based on deep learning. Nevise is available in two versions. The second version has greater accuracy, the ability to correct errors based on spaces, and a better understanding of special characters like half space. These versions can be accessed via web services and as demos. We provide public access to the code and model checkpoint of the first version here.

## packages Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install packages.

```bash
pip install -r requirements.txt
```
## Download model checkpoint and vocab and put them on "model" directory


```bash
pip install gdown
mkdir model
cd model
gdown https://drive.google.com/uc?id=1Ki5WGR4yxftDEjROQLf9Br8KHef95k1F
gdown https://drive.google.com/uc?id=1nKeMdDnxIJpOv-OeFj00UnhoChuaY5Ns
```
## run


```bash
python main.py
```
# Demo

[Nevise(both versions)](https://dadmatech.ir/#/products/SpellChecker)
