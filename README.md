# Mémoire

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

### To only see the results

#### Of all the test cases
__TAKES A FEW HOURS__

~~~~
python -m unittest discover
~~~~

#### Of a specific test

~~~
python -m unittest test.test.test.NameOfTheTest
~~~