import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-json', help='json file to sort')
parser.add_argument('-o', '--output-json', help='json file to sort')

args = parser.parse_args()

js = json.load(open(args.input_json))
with open(args.output_json, 'w') as f:
    json.dump(js, f, indent=4, sort_keys=True)