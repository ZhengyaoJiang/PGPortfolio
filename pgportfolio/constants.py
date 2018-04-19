#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import path

DATABASE_DIR = path.realpath(__file__).\
    replace('pgportfolio/constants.pyc','/database/Data.db').\
    replace("pgportfolio\\constants.pyc","database\\Data.db").\
    replace('pgportfolio/constants.py','/database/Data.db').\
    replace("pgportfolio\\constants.py","database\\Data.db")
CONFIG_FILE_DIR = 'net_config.json'
LAMBDA = 1e-4  # lambda in loss function 5 in training

# Time constants
NOW = 0
MINUTE = 60
THREE_MINUTE = MINUTE * 3
FIVE_MINUTE = MINUTE * 5
FIFTEEN_MINUTE = FIVE_MINUTE * 3
HALF_HOUR = FIFTEEN_MINUTE * 2
HOUR = HALF_HOUR * 2
TWO_HOUR = HOUR * 2
FOUR_HOUR = HOUR * 4
SIX_HOUR = HOUR * 6
EIGHT_HOUR = HOUR * 8
HALF_DAY = HOUR * 12
DAY = HOUR * 24
THREE_DAY = DAY * 3
WEEK = DAY * 7
MONTH = DAY * 30
YEAR = DAY * 365

# trading table name
TABLE_NAME = 'test'
