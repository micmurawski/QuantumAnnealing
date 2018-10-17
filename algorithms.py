import time
import math
import numpy as np
import random


def distance(point1, point2):
    return math.sqrt((point1[1] - point2[1]) ** 2 + (point1[0] - point2[0]) ** 2)


def spin_config_at_a_time_in_a_trotter_dim(tag, places):
    config = -np.ones(places, dtype=np.int)
    config[tag] = 1
    return config


def spin_config_in_a_trotter_dim(tag, places, init_config, total_time):
    spin = np.array([])
    spin = np.append(spin, init_config)
    for i in np.arange(total_time - 1):
        v = spin_config_at_a_time_in_a_trotter_dim(tag[i], places)
        spin = np.column_stack((spin, v))
    return spin


def get_spin_config(init_config, total_time, places, trotter_dim):
    spin = None
    for i in np.arange(trotter_dim):
        tag = np.arange(1, places)
        np.random.shuffle(tag)
        if spin is None:

            spin = spin_config_in_a_trotter_dim(tag, places, init_config, total_time)
        else:
            spin = np.dstack((spin_config_in_a_trotter_dim(tag, places, init_config, total_time), spin))
            # spin = np.append(spin,spin_config_in_a_trotter_dim(tag, places, init_config, total_time))

    return spin.T


# Trotter?
def get_best_route(config, points, total_time, trotter_dim, max_distance):
    length = np.array([])
    for i in np.arange(trotter_dim):
        route = np.array([])
        for j in np.arange(total_time):
            route = np.append(route, np.where(config[i][j] == 1)[0][0])
        length = np.append(length, get_total_distance(route, points, total_time, max_distance))

    min_tro_dim = np.argmin(length)
    best_route = np.array([])
    for i in np.arange(total_time):
        best_route = np.append(best_route, np.where(config[min_tro_dim][i] == 1)[0][0])
    return best_route


##
def get_total_distance(route, points, total_time, max_distance):
    total_distance = 0
    for i in np.arange(total_time):
        total_distance += distance(points[int(route[i])], points[int(route[(i + 1) % total_time])]) / max_distance
    return total_distance


##
def get_real_total_distance(route, points, total_time):
    total_distance = 0
    for i in np.arange(total_time):
        total_distance += distance(points[int(route[i])], points[int(route[(i + 1) % total_time])])
    return total_distance


##
def QMC_move(config, points, places, annealing_param, total_time, trotter_dim, beta, max_distance):
    #
    c = np.random.randint(0, trotter_dim)
    a_ = np.arange(1, total_time)
    a = np.random.choice(a_)
    a_ = np.delete(a_, a - 1)
    b = np.random.choice(a_)

    #
    p = np.where(config[c][a] == 1)[0][0]
    q = np.where(config[c][b] == 1)[0][0]

    #
    delta_cost = delta_costc = delta_costq_1 = delta_costq_2 = delta_costq_3 = delta_costq_4 = 0

    #
    for j in np.arange(places):
        l_p_j = distance(points[p], points[j]) / max_distance
        l_q_j = distance(points[q], points[j]) / max_distance
        delta_costc += 2 * (-l_p_j * config[c][a][p] - l_q_j * config[c][a][q]) * (
                config[c][a - 1][j] + config[c][(a + 1) % total_time][j]) + \
                       2 * (-l_p_j * config[c][b][p] - l_q_j * config[c][b][q]) * (
                               config[c][b - 1][j] + config[c][(b + 1) % total_time][j])

    #
    delta_costq_1 = config[c][a][p] * (config[(c - 1) % trotter_dim][a][p] + config[(c + 1) % trotter_dim][a][p])
    delta_costq_2 = config[c][a][q] * (config[(c - 1) % trotter_dim][a][q] + config[(c + 1) % trotter_dim][a][q])
    delta_costq_3 = config[c][b][p] * (config[(c - 1) % trotter_dim][b][p] + config[(c + 1) % trotter_dim][b][p])
    delta_costq_4 = config[c][b][q] * (config[(c - 1) % trotter_dim][b][q] + config[(c + 1) % trotter_dim][b][q])

    #
    para = (1 / beta) * math.log(
        math.cosh(beta * annealing_param / trotter_dim) / math.sinh(beta * annealing_param / trotter_dim))
    delta_cost = delta_costc / trotter_dim + para * (delta_costq_1 + delta_costq_2 + delta_costq_3 + delta_costq_4)

    if delta_cost <= 0:
        config[c][a][p] *= -1
        config[c][a][q] *= -1
        config[c][b][p] *= -1
        config[c][b][q] *= -1
    elif np.random.random() < np.exp(-beta * delta_cost):
        config[c][a][p] *= -1
        config[c][a][q] *= -1
        config[c][b][p] *= -1
        config[c][b][q] *= -1

    return config


def get_points(route, points):
    _points_x = np.array([])
    _points_y = np.array([])

    for idx in route:
        _points_x = np.append(_points_x, points[idx][0])
        _points_y = np.append(_points_y, points[idx][1])
    return _points_x, _points_y
