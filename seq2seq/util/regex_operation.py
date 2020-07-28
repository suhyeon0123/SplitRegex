#!/usr/bin/env python
# coding: utf-8

import FAdo
import re
from FAdo.reex import str2regexp


def pos_membership_test(regex, examples):    
    regex = str2regexp(regex)
    for word in examples:
        if regex.evalWordP(word):
            continue
        else:
            return False
    return True


def neg_membership_test(regex, examples):    
    regex = str2regexp(regex)
    for word in examples:
        if regex.evalWordP(word):
            return False
        else:
            continue
    return True


def regex_equal(regex1, regex2):
    dfa1 = str2regexp(regex1).toDFA()
    dfa2 = str2regexp(regex2).toDFA()
    return dfa1.equal(dfa2)


def preprocess_regex(regex1, regex2):
    regex1 = regex1.replace(' ','')
    regex1 = regex1.replace('[0-3]','(0|1|2|3)')
    regex2 = regex2.replace(' ','')
    regex2 = regex2.replace('[0-3]','(0|1|2|3)')
    return regex1, regex2


def regex_inclusion(target, predict):
    '''
    check if predict is belong to target 
    Let assume that target is superset of predict.
    '''
    target = str2regexp(target).toDFA()
    predict = str2regexp(predict).toDFA()
    
    if (target & predict).witness() != None and (~target & predict).witness() == None:
        return True
    else:
        return False
    
    
def valid_regex(regex):
    try:
        p = re.compile(regex)
        is_valid =True
    except re.error:
        is_valid=False
    return is_valid


def or_exception(regex):
    if '||' in regex or '|)' in regex or '(|' in regex or \
    regex[0] == '|' or regex[-1] =='|' or '()' in regex:
        return False
    else :
        return True
