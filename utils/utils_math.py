#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@project:project
@file: utils_math
@platform: PyCharm
@time: 2023/7/25
"""


def normalize(p):
    dim = p.size(-1)
    p.view(-1, dim).renorm_(2, 0, 1.)
    return p

def sqdist(p1, p2):
    return (p1 - p2).pow(2).sum(dim=-1)