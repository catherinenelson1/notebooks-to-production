# notebooks-to-production
PyData Seattle 2025 tutorial - "Going from Notebooks to Production Code"

- Download repo, install modules, run notebook
- Exercise 1: Modular Code. Decide how to break the code down into separate functions and implement them - either use our function stubs, partly filled in code, or completely DIY.
- Exercise 2: Unit Tests. Take your functions from Exercise 1 and write short tests to check that they work correctly. Run the tests using `pytest` and make sure that they pass! 


Files that won't be modified:
- penguins_data.csv: the data we'll use to train our model to predict penguin species
- penguins_notebook.ipynb: this is a notebook that explores the data, cleans it, trains a model, analyzes the model, and uses it to run inference on some new data. *This is what we're going to turn into production code.*
- penguins_simple_script.py: this is a script that was created by using a tool to convert the notebook into a script (with some minor tweaks). It does the same thing as the notebook and then outputs the model prediction for two specific new data points. *We won't modify this file, but it will be useful to ensure that as we write new code that our output agrees with this one.*
 
Files that we will modify:
- penguins_notebook_function_stubs.py: a skeleton consisting of functions to fill in for the different steps of the notebook (loading, cleaning, preprocessing, training, predicting)
- partial solution
- penguins_refactored.py: the complete solution for implementing the refactored version of the notebook
- partial unit tests file
- test_penguins_refactored.py: the complete set of unit tests

