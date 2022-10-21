from util import *
from collections import defaultdict




def data_partition(fname):
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    print('Preparing data...')
    f = open('data/%s.pkl' % fname)
    time_set = set()
    geo_set = []
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        u, i, rating, timestamp, lat, lon = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u] += 1
        item_count[i] += 1
    f.close()
    f = open('data/%s.pkl' % fname)
    for line in f:
        u, i, rating, timestamp, lat, lon = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        lat = float(lat)
        lon = float(lon)
        if user_count[u] < 5 or item_count[i] < 5:
            continue
        time_set.add(timestamp/60)
        geo_set.append([lat, lon])
        User[u].append([i, timestamp/60, lat, lon])
    f.close()
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    print('Preparing done...')

    return [user_train, user_valid, user_test, usernum, itemnum, timenum]



def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u+1
    for i, item in enumerate(item_set):
        item_map[item] = i+1

    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])
    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]], x[2], x[3]], items))
    time_max = set()

    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                time_diff.add(time_list[i + 1] - time_list[i])
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1] - time_min) / time_scale) + 1), x[2], x[3]],
                                  items))
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))
    return User_res, len(user_set), len(item_set), max(time_max)


