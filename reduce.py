import json

path = "Results/tib-core/predictions_level_1_final_updated.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

new_list = []
for item in data:
    print(len(item['predictions']))
    # item['predictions'] = item['predictions'][:2]
    # new_list.append(item)
    
# with open("Results/tib-core/predictions_level_1_final_updated.json", "w", encoding="utf-8") as f:
#         json.dump(new_list, f, indent=4, ensure_ascii=False)
print('reduction terminer')