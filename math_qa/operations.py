"""Mathematical functions in MathQA dataset and their implementation"""

import math


def square_edge_by_area(x1):
    return math.sqrt(x1)


def multiply(x1, x2):
    return x1 * x2


def permutation(x1, x2):
    n, m = min(x1, x2), max(x1, x2)
    return math.factorial(int(m)) / math.factorial(int(m - n))


def triangle_perimeter(x1, x2, x3):
    return x1 + x2 + x3


def surface_rectangular_prism(x1, x2, x3):
    return 2 * (x1 * x2 + x1 * x3 + x2 * x3)


def reminder(x1, x2):
    if x2 == 0:
        return 0
    return int(x1) % int(x2)


def stream_speed(x1, x2):
    return (x1 - x2) / 2


def volume_rectangular_prism(x1, x2, x3):
    return x1 * x2 * x3


def speed(x1, x2):
    return x1 / x2


def power(x1, x2):
    # might need here
    return x1 ** min(x2, 5)
    # return x1 ** x2


def quadrilateral_area(x1, x2, x3):
    return x1 * (x2 + x3) / 2


def cube_edge_by_volume(x1):
    return x1 ** (1 / 3)


def subtract(x1, x2):
    return x1 - x2


def original_price_before_loss(x1, x2):
    return x2 * 100 / (100 + 1e-5 - x1)


def log(x1):
    return math.log(max(1e-5, x1), 2)


def sine(x1):
    return math.sin(x1)


def add(x1, x2):
    return x1 + x2


def square_area(x1):
    return x1 ** 2


def p_after_gain(x1, x2):
    return (1 + x1 / 100) * x2


def surface_cube(x1):
    return 6 * x1**2


def speed_in_still_water(x1, x2):
    return (x1 + x2) / 2


def original_price_before_gain(x1, x2):
    return x1 * 100 / (100 + x2)


def volume_cylinder(x1, x2):
    return math.pi * x1**2 * x2


def surface_cylinder(x1, x2):
    return 2 * math.pi * x1 * x2 + 2 * math.pi * x1 ** 2


def divide(x1, x2):
    if x2 != 0:
        return x1 / x2
    return 0


def rectangle_area(x1, x2):
    return x1 * x2


def square_edge_by_perimeter(x1):
    return x1 / 4


def max(x1, x2):
    return x1 if x1 > x2 else x2


def cosine(x1):
    return math.cos(x1)


def min(x1, x2):
    return x1 if x1 < x2 else x2


def negate(x1):
    return -x1


def gcd(x1, x2):
    return math.gcd(int(x1), int(x2))


def negate_prob(x1):
    return 1 - x1


def choose(x1, x2):
    return math.comb(int(x1), int(x2))


def rectangle_perimeter(x1, x2):
    return 2 * x1 + 2 * x2


def volume_cone(x1, x2):
    return math.pi * x1**2 * x2 / 3


def rhombus_perimeter(x1):
    return 4 * x1


def volume_cube(x1):
    return x1 ** 3


def volume_sphere(x1):
    return 4 / 3 * math.pi * x1**3


def square_perimeter(x1):
    return 4 * x1


def lcm(x1, x2):
    q = abs(x1 * x2)
    if q == 0:
        return 0
    return q // math.gcd(int(x1), int(x2))


def rhombus_area(x1, x2):
    return x1 * x2 / 2


def diagonal(x1, x2):
    return math.sqrt(x1 ** 2 + x2 ** 2)


def triangle_area_three_edges(x1, x2, x3):
    s = x1 + x2 + x3
    s /= 2
    return math.sqrt(max(0, s * (s - x1) * (s - x2) * (s - x3)))


def circle_area(x1):
    return math.pi * x1 ** 2


def circumface(x1):
    return 2 * math.pi * x1


def surface_sphere(x1):
    return 4 * math.pi * x1**2


def inverse(x1):
    if x1 == 0:
        return 0
    if x1 is None:
        x2 = 2
    return 1 / x1


def sqrt(x1):
    return math.sqrt(max(0, x1))


def triangle_area(x1, x2):
    return x1 * x2 / 2


def tangent(x1):
    return math.tan(x1)


def factorial(x1):
    return math.factorial(min(15, abs(int(x1))))


def floor(x1):
    return math.floor(x1)


