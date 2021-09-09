#! /usr/bin/env python3
# UNTESTED CODE
import os
import os.path as path
import glob
import subprocess as sp
import time
import json

INPUT_PATH = 'inputs'
OUTPUT_PATH = 'outputs-{}'.format(time.strftime(
    '%m%d%H%M%S'))
RESULT_TXT = 'results.txt'

# examples:
# [ Example(pos, neg), ... ] or
# { "id": Example(pos, neg), ... }
def execute(examples,
            result_txt=RESULT_TXT,
            input_path=INPUT_PATH,
            output_path=OUTPUT_PATH):
    if type(examples) == list:
        examples = dict(zip(map(str, range(len(examples))), examples))

    if not path.exists(INPUT_PATH):
        os.makedirs(INPUT_PATH, exist_ok=True)

    if not path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH, exist_ok=True)
    print('aa')
    print(examples)
    preprocess_input_files(INPUT_PATH, examples)
    print('ff')
    return run_reggen(INPUT_PATH, OUTPUT_PATH, examples.keys())
    # results = read_results(OUTPUT_PATH)
    #
    # with open(RESULT_TXT, 'w') as f:
    #     for k, v in results.items():
    #         print('{}\t{}\t{}\t{}'.format(k, v['regex'], v['regexJS'], v['time']))
    #         f.write('{}\t{}\t{}\t{}\n'.format(k, v['regex'], v['regexJS'], v['time']))
    #


def preprocess_input_files(path, examples):
    for k, v in examples.items():
        item = {"name": k, "examples": []}

        for p in v.pos:
            item['examples'].append({
                'string': p,
                'match': [{
                    'start': 0,
                    'end': len(p)
                }],
                'unmatch': []
            })
        for n in v.neg:
            item['examples'].append({
                'string': n,
                'match': [],
                'unmatch': [{
                    'start': 0,
                    'end': len(n)
                }]
            })

        with open(path + '/{}.json'.format(k), 'w') as f:
            f.write(json.dumps(item))


def run_reggen(input_path, output_path, example_ids):
    COMMAND = [
        '/bin/sh',
        'submodels/RegexGenerator/regexturtle.sh',
        # 'regexturtle.sh',
        '-t', '1',  # threads (default 4)
        '-p', '100',  # GA population size (default 500)
        '-g', '100',  # max generations (default 100)
        '-e', '5.0',  # early stopping ratio to -g (default 20)
        '-o', output_path,
        '-d'  # dataset name; to be appended.
    ]
    for k in example_ids:
        ret = sp.run(COMMAND + [input_path + '/{}.json'.format(k)],
                stdout=sp.PIPE)
        for line in ret.stdout.split(b'\n'):
            if line.startswith(b'Best: '):
                regex = line[11:-4]  # discard ANSI escape seq
                try:
                    print('{}\t{}'.format(k, regex.decode()))
                    return regex.decode()
                except:
                    print(k, "[bytes->utf8 decode error]")
                break


def read_results(output_path):
    results = dict()
    for res in glob.glob(output_path + '/results-*.json'):
        with open(res, 'r') as f:
            item = json.loads(f.read())
        key = item['datasetName']
        regex = item['bestSolution']['solution']
        regexJS = item['bestSolution']['solutionJS']
        elapsed = item['overallExecutionTimeMillis']

        results[key] = {'regex': regex, 'regexJS': regexJS, 'time': int(elapsed)}

    return results


if __name__ == '__main__':
    class Ex():
        def __init__(self, pos, neg):
            self.pos = pos
            self.neg = neg

    print(execute([Ex(['000', '0000', '00000'], ['11', '1111', '11111'])]))
    print(execute([Ex(['6'], ['950561406', '681411297', '882', '84', '354380', '1498', '351721', '381', '994062614', '5480'])]))
