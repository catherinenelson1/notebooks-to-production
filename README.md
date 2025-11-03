# notebooks-to-production
PyData Seattle 2025 tutorial - "Going from Notebooks to Production Code"

- Download repo, install modules, run notebook
- Exercise 1: Modular Code. Decide how to break the code down into separate functions and implement them - either use our function stubs, partly filled in code, or completely DIY.
- Exercise 2: Unit Tests. Take your functions from Exercise 1 and write short tests to check that they work correctly. Run the tests using `pytest` and make sure that they pass! 


Top directory:
- penguins_data.csv: the data we'll use to train our model to predict penguin species
- notebook.ipynb: this is a notebook that explores the data, cleans it, trains a model, analyzes the model, and uses it to run inference on some new data. *This is what we're going to turn into production code.*
- simple_script.py: this is a script that was created by using a tool to convert the notebook into a script (with some minor tweaks). It does the same thing as the notebook and then outputs the model prediction for two specific new data points. *We won't modify this file, but it will be useful to ensure that as we write new code that our output agrees with this one.*
 
Answers folder:
- modular_code_function_stubs.py: a starting point for doing all the refactoring in Exercise 1
- partial solution for Exercise 1
- modular_code_full_solution.py: the complete solution for Exercise 1
- partial solution for Exercise 2
- unit_test_full_solution.py: the complete solution for Exercise 2

