#!/usr/bin/python3
import json

input = '/bigdata/zzf/spec_cpt/simpoint_summary.json'
max_point_per_workload = 20
target_coverage = 0.8
output = f"./simpoint_coverage{target_coverage}_test.json"

with open(input, 'r') as f:
  dict = json.load(f)
  new_dict = {}
  total_points = 0
  for workload in dict.keys():
    workload_dict = dict[workload]
    sorted_workload=sorted(workload_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

    weight_sum = 0
    point_num = 0
    new_dict[workload] = {}
    for point in sorted_workload:
      new_dict[workload][point[0]] = point[1]
      point_num += 1
      weight_sum += float(point[1])
      total_points += 1
      # print(workload, point, "added into dict, current total weight ", weight_sum)
      if (point_num > max_point_per_workload or weight_sum > target_coverage):
        break
    print(f"{workload} has {point_num} point{'s' if point_num > 1 else ''}, weight is {weight_sum}")
    # print(workload)
  # print(new_dict)
  print(total_points)
  with open(output, "w") as outf:
    outf.write(json.dumps(new_dict, indent=4))