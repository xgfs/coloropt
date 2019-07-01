from colormath import color_diff
from colormath.color_objects import sRGBColor, LabColor, HSVColor, CMYKColor, LCHabColor
from colormath.color_conversions import convert_color
import numpy as np
import itertools

black_lab = convert_color(sRGBColor(0, 0, 0), LabColor)
white_lab = convert_color(sRGBColor(1, 1, 1), LabColor)

def to_grayscale(color):
    if type(color) != sRGBColor:
        color_rgb = convert_color(color, sRGBColor)
    else:
        color_rgb = color
    r, g, b = color_rgb.get_value_tuple()
    gray_level = 0.21*r + 0.72*g + 0.07*b
    gray_srgb = sRGBColor(gray_level, gray_level, gray_level)
    return gray_srgb if type(color) == sRGBColor else convert_color(gray_srgb, type(color))

def clamp(color):
    if type(color) != sRGBColor:
        color_rgb = convert_color(color, sRGBColor)
    else:
        color_rgb = color
    rgb = np.array(color_rgb.get_value_tuple())
    rgb = np.clip(rgb, 0, 1)
    clamped_srgb = sRGBColor(*rgb)
    return clamped_srgb if type(color) == sRGBColor else convert_color(clamped_srgb, type(color))

def to_colorblind_g(color):
    if type(color) != sRGBColor:
        color_rgb = convert_color(color, sRGBColor)
    else:
        color_rgb = color
    r, g, b = color_rgb.get_upscaled_value_tuple()
    r_ = np.power(4211.106+0.6770*(g**2.2)+0.2802*(r**2.2), 1/2.2)
    g_ = np.power(4211.106+0.6770*(g**2.2)+0.2802*(r**2.2), 1/2.2)
    b_ = np.power(4211.106+0.95724*(b**2.2)+0.02138*(g**2.2)-0.02138*(r**2.2), 1/2.2)
    gray_srgb = sRGBColor(r_, g_, b_, True)
    return gray_srgb if type(color) == sRGBColor else convert_color(gray_srgb, type(color))

def to_colorblind_r(color):
    if type(color) != sRGBColor:
        color_rgb = convert_color(color, sRGBColor)
    else:
        color_rgb = color
    r, g, b = color_rgb.get_upscaled_value_tuple()
    r_ = np.power(782.74+0.8806*(g**2.2)+0.1115*(r**2.2), 1/2.2)
    g_ = np.power(782.74+0.8806*(g**2.2)+0.1115*(r**2.2), 1/2.2)
    b_ = np.power(782.74+0.992052*(b**2.2)-0.003974*(g**2.2)+0.003974*(r**2.2), 1/2.2)
    gray_srgb = sRGBColor(r_, g_, b_, True)
    return gray_srgb if type(color) == sRGBColor else convert_color(gray_srgb, type(color))

def window_stack(a, stepsize=1, width=3):
    n = a.shape[0]
    return np.hstack( a[i:1+n+i-width:stepsize] for i in range(0, width))

def anglediff(h1, h2):
    x, y = h1*np.pi/180, h2*np.pi/180
    return np.abs(np.arctan2(np.sin(x-y), np.cos(x-y))) * 180 / np.pi

def avg_cost(costs):
    n = costs.shape[0]
    avg_costs = []
    for window in range(2, n):
        window_costs = []
        for i in range(n-window):
            window_costs.extend(window_stack(costs[i, i+1:], 1, window).reshape(-1, window).sum(axis=1))
        avg_costs.append(np.mean(window_costs)/window)
    if not avg_costs:
        return 1
    return 1 - np.mean(avg_costs)

def multicolor_cost(colors, weights):    
    return np.sum(multicolor_cost_debug(colors, weights))/np.sum(weights)

def multicolor_cost_debug(colors, weights):
    scores = np.zeros(31)
    ncolors = len(colors)
    weights = np.array(weights)
    
    colors_lab = []
    for color in colors:
        colors_lab.append(convert_color(color, LabColor))
            
    cdists = np.zeros((ncolors, ncolors))
    for i in range(ncolors):
        for j in range(i+1, ncolors):
            dist = color_diff.delta_e_cie2000(colors_lab[i], colors_lab[j]) / 116
            cdists[i, j] = dist
            cdists[j, i] = dist
    
    quantiles = np.quantile(cdists[~np.eye(ncolors, dtype=bool)], [0, 0.25, 0.5, 0.75, 1])
    scores[0:5] = weights[0:5]*quantiles
    
    colors_lch = []
    for color in colors:
        colors_lch.append(convert_color(color, LCHabColor))

    cdists = np.zeros((ncolors, ncolors))
    for i in range(ncolors):
        for j in range(i+1, ncolors):
            dist = anglediff(colors_lch[i].lch_h, colors_lch[j].lch_h)
            cdists[i, j] = dist
            cdists[j, i] = dist

    reals = np.quantile(cdists[~np.eye(ncolors, dtype=bool)], [0, 0.25, 0.5, 0.75, 1])
    opts = np.array([2/ncolors, 0.25, 0.5, 0.75, 1])*360/2
    scores[5:10] = weights[5:10]*(1-np.abs(opts-reals)/opts)
        
    colors_hsv = []
    for color in colors:
        colors_hsv.append(convert_color(color, HSVColor))
                
    if weights[10] > 0 or weights[11] > 0:
        min_dist = 1000
        for color_lab in colors_lab:
            dist = color_diff.delta_e_cie2000(color_lab, white_lab) / 100
            if dist < min_dist:
                min_dist = dist
            scores[11] += weights[11] * dist / ncolors
        scores[10] = weights[10] * min_dist
    
    if weights[12] > 0 or weights[13] > 0:
        min_dist = 1000
        for color_lab in colors_lab:
            dist = color_diff.delta_e_cie2000(color_lab, black_lab) / 100
            if dist < min_dist:
                min_dist = dist
            scores[13] += weights[13] * dist / ncolors
        scores[12] = weights[12] * min_dist
            
    colors_gray = []
    for color_lab in colors_lab:
        colors_gray.append(to_grayscale(color_lab))
    
    if np.any(weights[14:19]>0):
        cdists = np.zeros((ncolors, ncolors))
        for i in range(ncolors):
            for j in range(i+1, ncolors):
                dist = color_diff.delta_e_cie2000(colors_gray[i], colors_gray[j]) / 116
                cdists[i, j] = dist
                cdists[j, i] = dist

        quantiles = np.quantile(cdists[~np.eye(ncolors, dtype=bool)], [0, 0.25, 0.5, 0.75, 1])
        scores[14:19] = weights[14:19]*quantiles
                
    if weights[19] > 0 or weights[20] > 0:
        min_dist = 1000
        for color_lab in colors_gray:
            dist = color_diff.delta_e_cie2000(color_lab, white_lab) / 100
            if dist < min_dist:
                min_dist = dist
            scores[20] += weights[20] * dist / ncolors
        scores[19] = weights[19] * min_dist
    
    if np.any(weights[21:26]>0):
        colors_cb_g = []
        for color_lab in colors_lab:
            colors_cb_g.append(to_colorblind_g(color_lab))
            
        cdists = np.zeros((ncolors, ncolors))
        for i in range(ncolors):
            for j in range(i+1, ncolors):
                dist = color_diff.delta_e_cie2000(colors_cb_g[i], colors_cb_g[j]) / 116
                cdists[i, j] = dist
                cdists[j, i] = dist

        quantiles = np.quantile(cdists[~np.eye(ncolors, dtype=bool)], [0, 0.25, 0.5, 0.75, 1])
        scores[21:26] = weights[21:26]*quantiles
    
    if np.any(weights[26:31]>0):
        colors_cb_r = []
        for color_lab in colors_lab:
            colors_cb_r.append(to_colorblind_r(color_lab))
            
        cdists = np.zeros((ncolors, ncolors))
        for i in range(ncolors):
            for j in range(i+1, ncolors):
                dist = color_diff.delta_e_cie2000(colors_cb_r[i], colors_cb_r[j]) / 116
                cdists[i, j] = dist
                cdists[j, i] = dist

        quantiles = np.quantile(cdists[~np.eye(ncolors, dtype=bool)], [0, 0.25, 0.5, 0.75, 1])
        scores[26:31] = weights[26:31]*quantiles
    
    return scores