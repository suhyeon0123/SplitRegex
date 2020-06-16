#!/usr/bin/env python
# coding: utf-8

# In[1]:
import FAdo
from FAdo.reex import str2regexp


def membership_test(regex, examples):    
    regex = str2regexp(regex)
    for word in examples:
        if regex.evalWordP(word):
            continue
        else:
            return False
    return True


def regex_equal(regex1, regex2):
    dfa1 = FAdo.reex.str2regexp(regex1).toDFA()
    dfa2 = FAdo.reex.str2regexp(regex2).toDFA()
    return dfa1.equal(dfa2)


def preprocess_regex(regex1, regex2):
    regex1 = regex1.replace(' ','')
    regex1 = regex1.replace('[0-3]','(0|1|2|3)')
    regex2 = regex2.replace(' ','')
    regex2 = regex2.replace('[0-3]','(0|1|2|3)')
    return regex1, regex2


def regex_inclusion(target, predict):
    '''
    check if predict is bre
    Let assume that target is superset of predict.
    '''
    target = str2regexp(target).toDFA()
    predict = str2regexp(predict).toDFA()
    intersection = ~target & predict
    if intersection.witness() == None:
        return True
    else:
        return False    
