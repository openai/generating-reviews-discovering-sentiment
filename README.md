# generating-reviews-discovering-sentiment

This is a fork of the code related to the paper [Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/abs/1704.01444) (Alec Radford, Rafal Jozefowicz, Ilya Sutskever).

It supports two use major use cases.

## Use the language model to generate reviews:
```
from encoder import Model
mdl = Model()

review_completed = mdl.generate_sequence("I couldn’t figure out", override={2388 : 1.0})
```
Optional constraints for `generate_sequence` shape the review to its desired length, tone and variance. For example, the parameters above complete the phrase with a postive sentiment by fixing the [infamous neuron 2388](https://blog.openai.com/unsupervised-sentiment-neuron/#sentimentneuron) to 1.0.

Stochastic and/or argmax sampling are used to vary the completed phrase. For an explanation of all parameters refer to the docstring and for example usage refer to `test_generate.py`. 

Some reviews that have been generated:
![alt text](https://raw.githubusercontent.com/ahirner/generating-reviews-discovering-sentiment/master/samples_sentiment.jpeg)


## Use the language model as feature extractor:
```
from encoder import Model

model = Model()
text = ['demo!']
text_features = model.transform(text)
```
