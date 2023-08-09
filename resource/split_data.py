from all_functions import open_json, write_json # frequently used functions
from parameters import RESTRUCTURED_DATA, TRAIN_DATA_FILE # parameters of the project
from parameters import TEST_DATA_FILE, VAL_DATA_FILE # parameters of the project

# open data and split it
data = open_json(RESTRUCTURED_DATA)
train = dict(list(data.items())[:11100])
test = dict(list(data.items())[11100:13400])
val = dict(list(data.items())[13400:])

# save data
write_json(TRAIN_DATA_FILE, train)
write_json(TEST_DATA_FILE, test)
write_json(VAL_DATA_FILE, val)
print('DONE!')