"""Constants used across the propagation models.

Includes neighborhood geometry, physical coefficients for wind/slope and
moisture effects, as well as fire-spotting and intensity parameters.
"""

import numpy as np
from numpy import array, pi

TICK_PRECISION = 10
CELLSIZE = 20

D1 = 0.5
D2 = 1.4
D3 = 8.2
D4 = 2.0
D5 = 50.0
A = 1 - ((D1 * (D2 * np.tanh((0 / D3) - D4))) + (0 / D5))

NEIGHBOURS = np.array(
    [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
)
NEIGHBOURS_DISTANCE = np.array([1.414, 1, 1.414, 1, 1, 1.414, 1, 1.414])
NEIGHBOURS_ANGLE = np.array(
    [
        pi * 3 / 4,
        pi / 2,
        pi / 4,
        pi,
        0,
        -pi * 3 / 4,
        -pi / 2,
        -pi / 4,
    ]
)


# aggiunte per spotting
# insieme punti lontani 2 celle
NEIGHBOURS_AT2 = [
    (-2, -2),
    (-2, -1),
    (-2, 0),
    (-2, 1),
    (-2, 2),
    (-1, -2),
    (-1, 2),
    (0, -2),
    (0, 2),
    (1, -2),
    (1, 2),
    (2, -2),
    (2, -1),
    (2, 0),
    (2, 1),
    (2, 2),
]
NEIGHBOURS_AT2_ARRAY = array(NEIGHBOURS_AT2)
NEIGHBOURS_AT2_DISTANCE = array(
    [
        [2.828, 2.236, 2, 2.236, 2.828],
        [2.236, 1.414, 1, 1.414, 2.236],
        [2, 1, 0, 1, 2],
        [2.236, 1.414, 1, 1.414, 2.236],
        [2.828, 2.236, 2, 2.236, 2.828],
    ]
)
NEIGHBOURS_AT2_ANGLE = array(
    [
        [pi * 3 / 4, pi * 13 / 20, pi / 2, pi * 7 / 20, pi / 4],
        [pi * 17 / 20, pi * 3 / 4, pi / 2, pi / 4, pi * 3 / 20],
        [pi, pi, np.nan, 0, 0],
        [pi * 17 / 20, pi * 3 / 4, pi / 2, pi / 4, pi * 3 / 20],
        [-pi * 3 / 4, -pi * 13 / 20, -pi / 2, -pi * 7 / 20, -pi / 4],
    ]
)
# insieme punti lontani 3 celle
NEIGHBOURS_AT3 = [
    (-3, -3),
    (-3, -2),
    (-3, -1),
    (-3, 0),
    (-3, 1),
    (-3, 2),
    (-3, 3),
    (-2, -3),
    (-2, 3),
    (-1, -3),
    (-1, 3),
    (0, -3),
    (0, 3),
    (1, -3),
    (1, 3),
    (2, -3),
    (2, 3),
    (3, -3),
    (3, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
]
NEIGHBOURS_AT3_ARRAY = array(NEIGHBOURS_AT3)

NEIGHBOURS_AT3_DISTANCE = array(
    [
        [4.243, 3.606, 3.162, 3, 3.162, 3.606, 4.243],
        [3.606, 2.828, 2.236, 2, 2.236, 2.828, 3.606],
        [3.162, 2.236, 1.414, 1, 1.414, 2.236, 3.162],
        [3, 2, 1, 0, 1, 2, 3],
        [3.162, 2.236, 1.414, 1, 1.414, 2.236, 3.162],
        [3.606, 2.828, 2.236, 2, 2.236, 2.828, 3.606],
        [4.243, 3.606, 3.162, 3, 3.162, 3.606, 4.243],
    ]
)
NEIGHBOURS_AT3_ANGLE = array(
    [
        [
            pi * 3 / 4,
            pi * 7 / 10,
            pi * 3 / 5,
            pi / 2,
            pi * 2 / 5,
            pi * 3 / 10,
            pi / 4,
        ],
        [
            pi * 4 / 5,
            pi * 3 / 4,
            pi * 13 / 20,
            pi / 2,
            pi * 7 / 20,
            pi / 4,
            pi / 5,
        ],
        [
            pi * 9 / 10,
            pi * 17 / 20,
            pi * 3 / 4,
            pi / 2,
            pi / 4,
            pi * 3 / 20,
            pi / 10,
        ],
        [pi, pi, pi, np.nan, 0, 0, 0],
        [
            -pi * 9 / 10,
            -pi * 17 / 20,
            -pi * 3 / 4,
            -pi / 2,
            -pi / 4,
            -pi * 3 / 20,
            -pi / 10,
        ],
        [
            -pi * 4 / 5,
            -pi * 3 / 4,
            -pi * 13 / 20,
            -pi / 2,
            -pi * 7 / 20,
            -pi / 4,
            -pi / 5,
        ],
        [
            -pi * 3 / 4,
            -pi * 7 / 10,
            -pi * 3 / 5,
            -pi / 2,
            -pi * 2 / 5,
            -pi * 3 / 10,
            -pi / 4,
        ],
    ]
)
# costante per calcolare distanza in fire-spotting
FIRE_SPOTTING_DISTANCE_COEFFICIENT = 0.191

# parametri Rothermel
ROTHERMEL_ALPHA1 = 0.0693
ROTHERMEL_ALPHA2 = 0.0576
# parametri Wang
WANG_BETA1 = 0.1783
WANG_BETA2 = 3.533
WANG_BETA3 = 1.2

# costanti per moisture
# probabilità
M1 = -3.5995
M2 = 5.2389
M3 = -2.6355
M4 = 1.019
# RoS
C_MOIST = -0.014

# The following constants are used in the Fire-Spotting model.
# Alexandridis et al. (2009,2011)

LAMBDA_SPOTTING = 2.0
SPOTTING_RN_MEAN = 100
SPOTTING_RN_STD = 25
# P_c = P_c0 (1 + P_cd), where P_c0 constant spread_probability of
# ignition by spotting and P_cd is a correction factor that
# depends on vegetation type and density...
P_C0 = 0.6

# variable for fireline intensity
Q = 2442.0


# --- FUEL SYSTEM LEGACY ---
NO_FUEL = 0


FUEL_SYSTEM_LEGACY_DICT = {
    1: dict(
        name="broadleaves",
        v0=140,
        d0=1.5,
        d1=3,
        hhv=20000,
        humidity=60,
        spread_probability={
            1: 0.3,
            2: 0.375,
            3: 0.005,
            4: 0.45,
            5: 0.225,
            6: 0.25,
            7: 0.075,
        },
    ),
    2: dict(
        name="shrubs",
        v0=140,
        d0=1,
        d1=3,
        hhv=21000,
        humidity=45,
        spread_probability={
            1: 0.375,
            2: 0.375,
            3: 0.005,
            4: 0.475,
            5: 0.325,
            6: 0.25,
            7: 0.1,
        },
    ),
    3: dict(
        name="non-vegetated",
        v0=20,
        d0=0.1,
        d1=0,
        hhv=100,
        humidity=-9999,
        spread_probability={
            1: 0.005,
            2: 0.005,
            3: 0.005,
            4: 0.005,
            5: 0.005,
            6: 0.005,
            7: 0.005,
        },
        burn=False,
    ),
    4: dict(
        name="grassland",
        v0=120,
        d0=0.5,
        d1=0,
        hhv=17000,
        humidity=-9999,
        spread_probability={
            1: 0.25,
            2: 0.35,
            3: 0.005,
            4: 0.475,
            5: 0.1,
            6: 0.3,
            7: 0.075,
        },
    ),
    5: dict(
        name="conifers",
        v0=200,
        d0=1,
        d1=4,
        hhv=21000,
        humidity=55,
        spread_probability={
            1: 0.275,
            2: 0.4,
            3: 0.005,
            4: 0.475,
            5: 0.35,
            6: 0.475,
            7: 0.275,
        },
        spotting=True,
        prob_ign_by_embers=0.4,
    ),
    6: dict(
        name="agro-forestry areas",
        v0=120,
        d0=0.5,
        d1=2,
        hhv=19000,
        humidity=60,
        spread_probability={
            1: 0.25,
            2: 0.3,
            3: 0.005,
            4: 0.375,
            5: 0.2,
            6: 0.35,
            7: 0.075,
        },
    ),
    7: dict(
        name="non-fire prone forests",
        v0=60,
        d0=1,
        d1=2,
        hhv=18000,
        humidity=65,
        spread_probability={
            1: 0.25,
            2: 0.375,
            3: 0.005,
            4: 0.475,
            5: 0.35,
            6: 0.25,
            7: 0.075,
        },
    ),
}
