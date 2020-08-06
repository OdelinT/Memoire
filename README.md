# Mémoire: environment simulation server 

Written in markdown. To read it more easily, you can convert it to docx with pandoc.

~~~
pandoc Memoire.md -t docx -o memoire.docx
~~~

## Install on windows

Requirements: python>=3.7.1 and <=3.8.5, pip, venv

~~~~
python -m venv env
.\env\Script\activate
pip install -r requirements.txt
python app.py
~~~~

## How to test

### To analyse exceptions

~~~~
pytest -q .\test\test.py
~~~~

### To see the results of the test cases

~~~~
python -m unittest discover
~~~~


## TODO
### Agent
- Define a protocol
- Use multiple algorithms
- Apply them to the environment

How to create a python environment:
- https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial#agent
- https://www.tensorflow.org/agents/tutorials/2_environments_tutorial

How to train it:
- https://www.tensorflow.org/agents/tutorials/0_intro_rl

### Environment
#### User stories
- Add price parameter
- Add price elasticity
- Places characteristics: 
  - Interest for the product (hidden)
  - Floor & ceiling prices
  - Parameters evolution over time
- Different products
  - For the same or different need

#### Technical features
- Add unit tests
- Add a configuration file for parameters enablement
- Add swagger
- Add OpenAPI auto documentation

#### Mandatory
- Ability to configure the environment programmatically to create complex situations

#### Already added
- Different places
- Places characteristic: indicative number of potential clients (given in get/places)
