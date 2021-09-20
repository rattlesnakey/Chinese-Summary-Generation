import json 
f = open('filter_sfzy_dataset.tsv', 'r')
o1 = open('train.json', 'w+')
o2 = open('dev.json', 'w+')
o3 = open('test.json', 'w+')
count = 0
for line in f:
    document, summary = line.strip().split('\t')
    temp_dict = {'document':document,'summary':summary}
    temp_json = json.dumps(temp_dict, ensure_ascii=False)
    if count <= 3850:
        o1.write(temp_json + '\n')
    if count >3850 and count < 3950:
        o2.write(temp_json + '\n')

    if count >= 3950:
        o3.write(temp_json + '\n')
    count += 1



    