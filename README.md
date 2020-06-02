# Mémoire: environment simulation server 

## Install on windows

Requirements : python 3.7+, pip, venv

~~~~
python -m venv env
.\env\Script\activate
pip install -r requirements.txt
python app.py
~~~~

## How to test
~~~~
python -m unittest discover -v
~~~~

## TODO
### User stories
- Add price parameter
- Add price elasticity
- Places characteristics: 
  - Interest for the product (hidden)
  - Floor & ceiling prices
  - Parameters evolution over time
- Different products
  - For the same or different need

### Technical features
- Add unit tests
- Add a configuration file for parameters enablement
- Add swagger
- Add OpenAPI auto documentation


## Already added
- Different places
- Places characteristic: indicative number of potential clients (given in get/places)
