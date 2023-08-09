import json
import pickle as pkl

# open JSON file
def open_json(file_path):
    with open(file_path) as f:
        content = json.load(f)
    return content

# close JSON file
def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return print(file_path + ': file is SAVED')

# Put IDS to tags
def put_id_to_tags(data):
    data_list = list(set(data))
    data_list.sort()
    data_dict = dict()
    i = 0
    for d in data_list:
        data_dict[d] = i
        i += 1
    return data_dict

# open features
def open_feats(file_path):
    with open(file_path, 'rb') as handle:
        feat_dict = pkl.load(handle)
    return feat_dict

# save features
def save_feats(file_path, data):
    with open(file_path, 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)    
    return 'File saved!'

def edit_feats(feats, labels):
    new_feats = dict()
    for f in feats:
        if str(f) in labels.keys():
            new_feats[f] = feats[f]
    return new_feats

def get_labels(feats, labels):
    Y = list()
    for f in feats:
        Y.append(labels[str(f)])
    return Y
