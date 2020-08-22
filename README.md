# Mémoire

## Abstract

### Français

L’inférence de causalité à partir de données est une question majeure en statistiques. La causalité est une part importante de la cognition humaine. De leur côté, les algorithmes d’apprentissage automatique sur lequel reposent les intelligences artificielles se basent essentiellement sur des corrélations, des liens statistiques entre des variables. Corrélation n’est pas causalité, et cette distinction est parfois présentée comme l’un des freins majeurs au développement des intelligences artificielles. Un type d’apprentissage automatique se distingue des autres sur la question. Il s’agit de l’apprentissage par renforcement, qui consiste à laisser un algorithme (un agent) apprendre en interagissant avec son environnement. En l’appliquant à des situations biaisées, on peut donc observer dans quelle mesure un algorithme est sensible à ces biais, donc dans quelle mesure cet algorithme est capable de différencier une corrélation d’une causalité. La première expérience a quantifié l’intérêt de trop contraindre un agent sur le court terme et le risque que cela occasionne dans certaines conditions sur le long terme. La seconde montre la difficulté d’un agent à appréhender une relation de causalité avec une variable qu’il ne peut observer lorsque cette dernière évolue.

### English

Causality inference from data is a major issue in statistics. Causality is an important part of human cognition. On the other hand, the automatic learning algorithms used for artificial intelligences are based are essentially based on correlations, statistical links between variables. Correlation is not causality, and this distinction is sometimes presented as one of the major obstacles to the development of artificial intelligence. One type of machine learning stands out from the others on the issue. This is reinforcement learning, which consists in letting an algorithm (an agent) learn by interacting with its environment. By applying it to biased situations, we can therefore observe how sensitive an algorithm is to these biases, and therefore to what extent it is able to differentiate a correlation from a causality. The first of the experiments quantified the benefit of over-constraining an agent in the short term, and the risk that this entails under certain conditions in the long term. The second shows the difficulty of an agent in understanding a causal relationship with a variable that it cannot observe when this variable evolves.

## The final dissertation

It is accessible in as a PDF in french [here](Memoire_Odelin_Tamayo.pdf).

The previous versions were written in french only in markdown. To read it more easily, you can convert it to docx with pandoc:

~~~
pandoc Memoire.md -t docx -o memoire.docx
~~~

## How to run the experiments

### Install on windows

Requirements: python>=3.7.1 and <=3.8.5, pip, venv

~~~~
python -m venv env
.\env\Script\activate
pip install -r requirements.txt
python app.py
~~~~

### How to test

#### To analyse exceptions

##### Of all the test cases - slow

~~~~
pytest -q .\test\test.py
~~~~

##### Of a specific test

~~~
pytest -q .\test\test.py::test::NameOfTheTest
~~~

#### To only see the results

##### Of all the test cases - slow

~~~
python -m unittest discover
~~~

##### Of a specific test

~~~
python -m unittest test.test.test.NameOfTheTest
~~~