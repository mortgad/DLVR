# DLVR
Deep Learning for Visual Recognition

Colab Github Demo:
https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=K-NVg7RjyeTk

## Workflow

### Notebooks

1: Navigate to notebook via GitHub
2: Open in Colab
  2a: If necessary, clone the repo from within the Colab view to use utils
3: Make changes, make temporary saves using "File" -> "Save a copy to Drive"
4: To push changes to GitHub, press "File" -> "Save a copy to GitHub"

### Python scripts

1: Use GitHub normally :)

## Cloning GitHub repo within a colab file

1: Clone repository
2: Navigate to the repository

'''
!git clone https://github.com/mortgad/DLVR.git
%cd DLVR
'''

Then, you can import from utils, e.g.:

'''
from utils.tokenizer import tokenize
tokenize(text="Hello World")
'''
