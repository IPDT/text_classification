#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/15/2018 9:53 AM
# @Author  : SkullFang
# @Contact : yzhang.private@gmail.com
# @File    : data_precess.py
# @Software: PyCharm
import json
fin=open('data/merge_2.txt','r')
fout=open('data/simple3.csv','w')
return_list=[]
fout.write('Category|Descript\n')
for line in fin.readlines():
    sub_str_list=line.split('\t',1)
    sub_label=sub_str_list[0]
    sub_description=sub_str_list[1]
    new_Str=sub_label+'|'+sub_description
    print(new_Str)
    fout.write(new_Str)