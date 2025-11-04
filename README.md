# Going from Notebooks to Production Code

A tutorial presented at PyData Seattle 2025 by Catherine Nelson and Robert Masson.

## Overview

- Download repo, install modules, run notebook
- Exercise 1: Modular Code. Decide how to break the code down into separate functions and implement them - either use our function stubs, partly filled in code, or completely DIY.
- Exercise 2: Unit Tests. Take your functions from Exercise 1 and write short tests to check that they work correctly. Run the tests using `pytest` and make sure that they pass!

## Installation


## Repository structure

Top level directory:
- `penguins_data.csv`: the data we'll use to train our model to predict penguin species
- `notebook.ipynb`: this is a notebook that explores the data, cleans it, trains a model, analyzes the model, and uses it to run inference on some new data. *This is what we're going to turn into production code.*

Exercises folder:

- `simple_script.py`: this is a script that was created by using a tool to convert the notebook into a script (with some minor tweaks). It does the same thing as the notebook and then outputs the model prediction for two specific new data points. *We won't modify this file, but it will be useful to ensure that as we write new code that our output agrees with this one.*
- `modular_code_function_stubs.py`: a starting point for Exercise 1. Choose this if you want to implement all the functions yourself.
- `modular_code_exercise.py`: another starting point for Exercise 1. Choose this if you want to implement 2 functions.
- `test_modular_code_exercise.py`: the starting point for Exercise 2.
 
Answers folder (don't look at this until you've done the exercises!):

- `modular_code_full_solution.py`: the complete solution for Exercise 1
- `test_full_solution.py`: the complete solution for Exercise 2

