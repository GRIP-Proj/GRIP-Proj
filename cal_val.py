import re
pattern = r"score\s*is\s*([0-9.+-eE]+)"

model_list = ['gpd', 'graspGen', 'graspnet']

for model in model_list:
    total_score = 0.0
    model_path = f'./output2_{model}.log'
    with open(model_path, 'r') as f:
        lines = f.readlines()
    validate_line = 0
    # print(len(lines))
    for line in lines:
        if line.startswith('score'):
            match = re.search(pattern, line)
            if match:
                validate_line += 1
                cur_score = float(match.group(1))
                total_score += cur_score if cur_score < 1 else 1.0
    print(f'model: {model}, total score: {total_score}, valid lines: {validate_line}')
    # break