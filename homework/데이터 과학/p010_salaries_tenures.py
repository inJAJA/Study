from collections import defaultdict

salaries_and_tenures = [(83000, 8.7),(88000, 8.1),
                        (48000, 0.7),(76000, 6),
                        (69000, 6.5),(76000, 7.5),
                        (60000, 2.5),(83000, 10),
                        (48000, 1.9),(63000, 4.2)]

# 키는 근속 연수, 값은 해당 근속 연수에 대한 연봉 목록
salary_by_tenure = defaultdict(list)

for salary , tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)

print(salary_by_tenure)
#defaultdict(<class 'list'>, {8.7: [83000], 8.1: [88000], 0.7: [48000],
#                               6: [76000], 6.5: [69000], 7.5: [76000], 
#                             2.5: [60000], 10: [83000], 1.9: [48000],
#                             4.2: [63000]})

# 키는 근속 연수, 값은 해당 근속 연수의 평균 연봉
average_salart_by_tenure = {
    tenure : sum(salaries) / len(salaries)
    for tenure, salaries in salary_by_tenure.items()
}

print(average_salart_by_tenure)
# {8.7: 83000.0, 8.1: 88000.0, 0.7: 48000.0, 6: 76000.0, 6.5: 69000.0, 
# 7.5: 76000.0, 2.5: 60000.0, 10: 83000.0, 1.9: 48000.0, 4.2: 63000.0}

def tenure_bucket(tenure):
    if tenure <2:
        return "less than two"
    elif tenure <5:
        return "between two and five"
    else:
        return "more than five"
    

# 키는 근속 연수 구간, 값은 해당 구간에 속하는 사용자들의 연봉
salary_by_tenure_bucket = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)

print(salary_by_tenure_bucket)
# defaultdict(<class 'list'>, 
# {'more than five': [83000, 88000, 76000, 69000, 76000, 83000], 
# 'less than two': [48000, 48000],
# 'between two and five': [60000, 63000]})


# 키는 근속 연수 구간, 값은 해당 구간에 속하는 사용자들의 평균 연봉
average_salart_by_bucket = {
    tenure_bucket: sum(salaries) / len(salaries)
    for tenure_bucket, salaries in salary_by_tenure_bucket.items()
}

print(average_salart_by_bucket)
#{'more than five': 79166.66666666667, 
# 'less than two': 48000.0, 
# 'between two and five': 61500.0}