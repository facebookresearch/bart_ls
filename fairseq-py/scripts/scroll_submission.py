from datasets import load_dataset
import json

def read_predictions(file):
    data = [l.strip() for l in open(file).readlines()]
    return data


# dataname = 'qmsum'
# split = 'test'
# dataset = load_dataset('tau/scrolls', dataname)

# samples = dataset[split]
# preds = read_predictions('/fsx/xwhan/data/scrolls/qmsum/test.best')
# results = {}
# assert len(samples) == len(preds)
# for item, pred in zip(samples, preds):
#     results[item['id']] = pred

# json.dump(results, open('/fsx/xwhan/data/scrolls/qmsum/test_sub.json', 'w'))



# dataname = 'quality'
# split = 'test'
# dataset = load_dataset('tau/scrolls', dataname)

# samples = dataset[split]
# preds = read_predictions('/fsx/xwhan/data/scrolls/quality/test.best')
# results = {}
# assert len(samples) == len(preds)
# for item, pred in zip(samples, preds):
#     results[item['id']] = pred

# json.dump(results, open('/fsx/xwhan/data/scrolls/quality/test_sub.json', 'w'))



# dataname = 'contract_nli'
# split = 'test'
# dataset = load_dataset('tau/scrolls', dataname)

# samples = dataset[split]
# preds = read_predictions(f'/fsx/xwhan/data/scrolls/{dataname}/test.best')
# results = {}
# assert len(samples) == len(preds)
# for item, pred in zip(samples, preds):
#     results[item['id']] = pred

# json.dump(results, open(f'/fsx/xwhan/data/scrolls/{dataname}/test_sub.json', 'w'))



dataname = 'narrative_qa'
split = 'test'
dataset = load_dataset('tau/scrolls', dataname)

samples = dataset[split]
preds = read_predictions(f'/fsx/xwhan/data/scrolls/{dataname}/test.best')
results = {}
assert len(samples) == len(preds)
for item, pred in zip(samples, preds):
    results[item['id']] = pred

json.dump(results, open(f'/fsx/xwhan/data/scrolls/{dataname}/test_sub.json', 'w'))