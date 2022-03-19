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

# Results on [Nevise Dataset](https://github.com/Dadmatech/Nevise-Dataset/tree/main/nevise-news-title-539)

</br>

| Algorithm | Wrong detection rate | Wrong correction rate | Correct to wrong rate | Precision |
| -- | -- | -- | -- | -- |
| Nevise 2 | **0.8314** | **0.7216** | 0.003 | 0.968 |
| Paknevis | 0.7843 | 0.6706 | 0.228 | 0.7921 |
| Nevise 1 | 0.7647 | 0.6824 | **0.0019** | **0.9774** |
| Google | 0.7392 | 0.702 | 0.0045 | 0.9449 |
| Virastman | 0.6 | 0.5 | 0.0032 | 0.9533 |
