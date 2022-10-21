from math import radians, cos, asin, sin, sqrt
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Process, Queue
import numpy as np

class LinerRegressionModel(object):
    def __init__(self, data):
        self.data = data
        self.x = data[:, 0]
        self.y = data[:, 1]

    def plt(self, a, b):
        plt.plot(self.x, self.y, 'o', label='data', markersize=10)
        plt.plot(self.x, a * self.x + b, 'r', label='line')
        plt.legend()




    def least_square_method(self):
        def calc_ab(x, y):
            sum_x, sum_y, sum_xy, sum_xx = 0, 0, 0, 0
            n = len(x)
            for i in range(0, n):
                sum_x += x[i]
                sum_y += y[i]
                sum_xy += x[i] * y[i]
                sum_xx += x[i]**2
            a = (sum_xy - (1/n) * (sum_x * sum_y)) / (sum_xx - (1/n) * sum_x**2)
            b = sum_y/n - a * sum_x/n
            return a, b
        a, b = calc_ab(self.x, self.y)
        # self.log(a, b)
        self.plt(a, b)
        return a, b
def get_fuction(user_train,time_mean):

    short_timelist=[]
    long_timelist=[]
    short_geolist=[]
    long_geolist=[]

    for user,item in user_train.items():
        user_timelist=[]


        user_geolist=[]
        for i in range(len(item)-1):
            user_timelist.append(item[i][1])
            user_geolist.append(haversine([item[i+1][2],item[i+1][3]],[item[i][2],item[i][3]]))
        user_meanlist=0
        for j in range(len(user_timelist)-1):
            user_meanlist+=user_timelist[j]
        user_mean=user_meanlist/len(user_timelist)
        if user_mean < time_mean:
            short_timelist.extend(user_timelist)
            short_geolist.extend( user_geolist)
        else:
            long_timelist.extend(user_timelist)
            long_geolist.extend( user_geolist)


    data_short = np.array(list(zip(short_timelist, short_geolist)))
    model = LinerRegressionModel(data_short)
    a_short, b_short = model.least_square_method()

    data_long = np.array(list(zip(long_timelist,long_geolist)))
    model = LinerRegressionModel(data_long)
    a_long, b_long = model.least_square_method()
    return  a_short, b_short , a_long, b_long



def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def get_gltimemean(user_train):
    usertime_mean = []
    user_timemax=[]
    for user, item in user_train.items():
        usertime = []
        for i in item:
            usertime.append(i[1])
        usertime_interval = []
        for j in range(len(usertime) - 1):
            usertime_interval.append(abs(usertime[j + 1] - usertime[j]))
        user_timemax.append(max(usertime_interval))
        user_intervalsum = 0
        for k in usertime_interval:
            user_intervalsum += k
        usertime_mean.append(round(user_intervalsum / len(usertime_interval)))
    timemean_sum = 0
    for time in usertime_mean:
        timemean_sum += time
    global gl_timemean
    gl_timemax=max(user_timemax)
    gl_timemean = round(timemean_sum / len(usertime_mean))
    print(gl_timemean,gl_timemax)
    return gl_timemean


def get_glgeomean(user_train):
    usergeo_mean = []
    user_geomax = []
    for user, item in user_train.items():
        usergeo = []

        for i in item:
            usergeo.append([i[2], i[3]])
        usergeo_interval = []

        for j in range(len(usergeo) - 1):
            usergeo_interval.append(haversine([item[j+1][2],item[j+1][3]],[item[j][2],item[j][3]]))
        user_geomax.append(max(usergeo_interval))
        user_intervalsum = 0
        for k in usergeo_interval:
            user_intervalsum += k
        usergeo_mean.append(round(user_intervalsum / len(usergeo_interval)))
    geomean_sum = 0
    for time in usergeo_mean:
        geomean_sum += time
    gl_geomean = round(geomean_sum / len(usergeo_mean))
    gl_geomax = max(user_geomax)
    return gl_geomean, gl_geomax


def computeRePos(time_seq, time_span):
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])

            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def Relation(user_train, usernum, maxlen, time_span):
    time_data_train = dict()
    for user in tqdm(range(1, usernum + 1), desc='Preparing time relation matrix'):
        # print(user)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            idx -= 1
            if idx == -1: break
        time_data_train[user] = computeRePos(time_seq, time_span)

    return time_data_train


def computeRePosgeo(geo_seq, allgeo_span, usertime_seq, time_span, a_short, b_short , a_long, b_long, gl_timemean):
    time_interval= []
    geo_interval=[]
    for i in range(len(usertime_seq)-1):
        time_interval.append(usertime_seq[i+1]-usertime_seq[i])
    user_time_mean=0
    for i in range(len(time_interval)-1):
        user_time_mean+=time_interval[i]
    user_mean=user_time_mean/len(time_interval)
    time_intervalmax=max(time_interval)
    for j in range(len(geo_seq) - 1):
        geo_interval.append(haversine([geo_seq[j+1][2],geo_seq[j+1][3]],[geo_seq[j][2],geo_seq[j][3]]))

    if user_mean < gl_timemean:
        geo_span=a_short*time_intervalmax+b_short
        if geo_span<allgeo_span:

            geo_matrix = np.zeros([len(geo_seq), len(geo_seq)], dtype=np.int32)
            for i in range(len(geo_seq)-1):
                for j in range(len(geo_seq)-1):
                    span = abs(haversine([geo_seq[j+1][2],geo_seq[j+1][3]],[geo_seq[j][2],geo_seq[j][3]]))
                    if span > geo_span:
                        geo_matrix[i][j] = geo_span
                    else:
                        geo_matrix[i][j] = span
        else:
            geo_matrix = np.zeros([len(geo_seq), len(geo_seq)], dtype=np.int32)
            for i in range(len(geo_seq) - 1):
                for j in range(len(geo_seq) - 1):
                    span = abs(haversine([geo_seq[j+1][2],geo_seq[j+1][3]],[geo_seq[j][2],geo_seq[j][3]]))
                    if span > allgeo_span:
                        geo_matrix[i][j] = allgeo_span
                    else:
                        geo_matrix[i][j] = span



    else:
        if user_mean == gl_timemean:
            number_eql = 1
        else:
            if user_mean < time_span:
                geo_span=a_long*time_intervalmax+b_long
                if geo_span<allgeo_span:


                    geo_matrix = np.zeros([len(geo_seq), len(geo_seq)], dtype=np.int32)
                    for i in range(len(geo_seq)-1):
                        for j in range(len(geo_seq)-1):
                            span = abs(haversine([geo_seq[j+1][2],geo_seq[j+1][3]],[geo_seq[j][2],geo_seq[j][3]]))
                            if span > geo_span:
                                geo_matrix[i][j] = geo_span
                            else:
                                geo_matrix[i][j] = span
                else:
                    geo_matrix = np.zeros([len(geo_seq), len(geo_seq)], dtype=np.int32)
                    for i in range(len(geo_seq) - 1):
                        for j in range(len(geo_seq) - 1):
                            span = abs(haversine([geo_seq[j+1][2],geo_seq[j+1][3]],[geo_seq[j][2],geo_seq[j][3]]))
                            if span > allgeo_span:
                                geo_matrix[i][j] = allgeo_span
                            else:
                                geo_matrix[i][j] = span
            else:
                geo_matrix = np.zeros([len(geo_seq), len(geo_seq)], dtype=np.int32)
                for i in range(len(geo_seq)-1):
                    for j in range(len(geo_seq)-1):
                        span = abs(haversine([geo_seq[j+1][2],geo_seq[j+1][3]],[geo_seq[j][2],geo_seq[j][3]]))
                        if span > allgeo_span:
                            geo_matrix[i][j] = allgeo_span
                        else:
                            geo_matrix[i][j] = span

    return geo_matrix


def Relation_geo(user_train, usernum, maxlen, geo_span,time_span,a_short, b_short , a_long, b_long, gl_timemean):
    geo_data_train = dict()
    number1=0
    for user in tqdm(range(1, usernum + 1), desc='Preparing geo relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        geo_seq = [[0, 0]] * maxlen
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            geo_seq[idx] = [i[2], i[3]]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        geo_data_train[user]= computeRePosgeo(geo_seq, geo_span, time_seq, time_span, a_short, b_short , a_long, b_long, gl_timemean)

    return geo_data_train


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, relation_time_matrix, relation_geo_matrix,
                    result_queue, SEED):
    def sample(user):

        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        geo_seq = [[0, 0]] * maxlen
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1][0]

        idx = maxlen - 1
        ts = set(map(lambda x: x[0], user_train[user]))
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            geo_seq[idx] = [i[2], i[3]]
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1: break
        time_matrix = relation_time_matrix[user]
        geo_matrix = relation_geo_matrix[user]
        return (user, seq, time_seq, time_matrix, geo_seq, geo_matrix, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1, usernum + 1)
            while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, relation_time_matrix, relation_geo_matrix, batch_size=128, maxlen=10,
                 n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      relation_time_matrix,
                                                      relation_geo_matrix,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def hav(theta):
    s = sin(theta / 2)
    return s * s



def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000



def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:
        time_map[time] = int(round(float(time - time_min)))

    return time_map



