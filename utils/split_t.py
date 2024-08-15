#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@project:project
@file: split_t
@platform: PyCharm
@time: 2023/7/11
"""

def split_df(all_df, args):

    smallest_t = min(all_df[2])
    largest_t = max(all_df[2])

    total_t = int(largest_t-smallest_t)
    offset = total_t/args.split_t_num
    t_list = range(args.split_t_num)
    all_df_t = []

    for t in t_list:
        s_t = smallest_t + offset * t
        if t != args.split_t_num-1:
            e_t =smallest_t + offset * (t+1)
        else:
            e_t = largest_t + 100
        df_t = all_df.loc[(all_df[2] >= s_t) & (all_df[2] < e_t)]
        all_df_t.append(df_t)

    return all_df_t, t_list