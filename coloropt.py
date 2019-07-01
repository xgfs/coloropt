import click
import logging
import uuid
import sys
import numpy as np

from scipy.optimize import minimize
from colortools import *
from colormath.color_objects import *


@click.command()
@click.argument('weights', type=int, nargs=31)
@click.argument('hues', type=int, nargs=-1)
@click.option('--c_from', type=float, default=1, help='a')
@click.option('--c_to', type=float, default=1, help='b')
@click.option('--h_from', type=float, default=0, help='c')
@click.option('--h_to', type=float, default=360, help='d')
@click.option('--l_from', type=float, default=1, help='a')
@click.option('--l_to', type=float, default=1, help='b')
@click.option('--logdir', type=str, default='logs', help='d')
def main(weights, hues, c_from, c_to, h_from, h_to, l_from, l_to, logdir):
    weights = np.array(weights)
    hues = np.array(hues)

    
    def cost_function(x):
        colors = []
        for l, c, h in zip(*[iter(x)]*3):
            if h > h_to or h < h_from:
                return 0
            if l > l_to or l < l_from:
                return 0
            if c > c_to or c < c_from:
                return 0
            colors.append(clamp(LCHabColor(l, c, h)))
        return -multicolor_cost(colors, weights)

    #  set up logging
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    run_id = uuid.uuid4().hex[:16]

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_filename = f'{logdir}/{run_id}.log'
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root.handlers = [file_handler, console_handler]

    root.info(f'Starting with parameters: weights={weights} hues={hues} c_from={c_from} c_to={c_to} h_from={h_from} h_to={h_to} l_from={l_from} l_to={l_to}')
    x0 = []
    for h in hues:
        x0.extend([(l_from+l_to)/2, (c_from+c_to)/2, h])
    res = minimize(cost_function, x0, method='Powell', tol=1e-9, options={'maxfev': len(x0)*10000, 'disp': True})
    colors = []
    for l, c, h in zip(*[iter(res.x)]*3):
        colors.append(convert_color(clamp(LCHabColor(l, c, h)), sRGBColor))
    root.info(f'Score={multicolor_cost(colors, weights)} colors: {list(map(lambda x: x.get_upscaled_value_tuple(), colors))}')            

    file_handler.close()
    console_handler.close()


if __name__ == '__main__':
    main()
