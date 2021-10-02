# coloropt

This code accompanies [my blogpost](http://tsitsul.in/blog/coloropt/) on color optimization.

### Basic usage

First, install the requirements:

    pip install -r requirements.txt

or simply

    pip install colormath numpy click

Then, you can use the command-line tool as follows:

    python coloropt.py WEIGHT_VECTOR INITIAL_HUES --c_from 0 --c_to 100 --l_from 0 --l_to 100 --logfir logs

where ``WEIGHT_VECTOR`` is a parameters for the objective function and ``INITIAL_HUES`` are initial guesses for the hue values for the optimal colors. Example usage:

    python coloropt.py 500 100 75 50 25 100 75 50 25 10 25 10 20 10 1000 100 50 25 10 30 10 350 150 100 50 25 200 100 50 25 10 40 200 240 360 --c_from 50 --c_to 75 --l_from 40 --l_to 75 --logdir logs
    
    
### License

You are free to use the pallettes and the code without attribution in commercial projects. I would love if you drop an email if you find them useful.
