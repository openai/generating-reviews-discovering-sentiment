# generating-reviews-discovering-sentiment
Code for "Learning to Generate Reviews and Discovering Sentiment"

This is where we'll be putting code related to the paper [Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/abs/1704.01444) (Alec Radford, Rafal Jozefowicz, Ilya Sutskever).

Right now the code supports using the language model as feature extractor.

```
from encoder import Model

model = Model()
text = ['demo!']
text_features = model.transform(text)
```
