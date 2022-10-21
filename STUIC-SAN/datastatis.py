import pandas as pd

from util import *


def Statistics(dataset,alljob_meangeolist,allfre_meangeolist):
    for user, items in dataset.items():
        user_job_time=0
        user_free_time = 0
        user_job_geo = 0
        user_free_geo=0
        job_num = 0
        free_num=0
        for i in range(len(items) - 1):
            if items[i][4] < 5 and items[i+1][4] < 5:
                if items[i][4] == items[i+1][4]:
                    user_job_time += abs(items[i+1][1]-items[i][1])
                    user_job_geo += round(haversine(items[i][2],items[i][3],items[i+1][2],items[i+1][3]))
                    job_num+=1

            if items[i][4] > 4 and items[i+1][4] > 4 :
                    user_free_time += abs(items[i+1][1]-items[i][1])
                    user_free_geo += round(haversine(items[i][2],items[i][3],items[i+1][2],items[i+1][3]))
                    free_num+=1

        if  free_num !=0 and job_num!=0:
            alljob_meangeolist.append(round(user_job_geo/job_num))
            allfre_meangeolist.append(round(user_free_geo / free_num))
    jobmean=round(sum(alljob_meangeolist)/len(alljob_meangeolist))
    fremean=round(sum(allfre_meangeolist)/len(allfre_meangeolist))
    mean=round((sum(alljob_meangeolist)/len(alljob_meangeolist)+sum(allfre_meangeolist)/len(allfre_meangeolist))/2)
    return jobmean,fremean,mean

def get_ts(data):
    mean_overall_pearson_r=0
    user_num=0
    for user, item in data.items():
        user_num+=1
        time_list=[]
        geo_list=[]
        for i in item:
            time_list.append(i[1])
            geo_list.append([i[2],i[3]])
        time_intervallist=[]
        time_interval=[]
        geo_intervallist=[]
        geo_interval=[]
        for i in time_list :
            time_intervallist.append(abs(time_list[i+1]-time_list[i]))
        for i in time_intervallist:
            time_interval.append(time_intervallist[i])

        for j in range(len(geo_list)-1) :
            geo_intervallist.append(haversine(geo_list[j][2],geo_list[j][3],geo_list[j+1][2],geo_list[j+1][3]))
        for i in range(len(geo_intervallist) - 1):
            geo_interval.append(geo_intervallist[i])
        df = pd.DataFrame(data=[time_interval, geo_interval], index=['timeinterval', 'geointerval']).T
        overall_pearson_r = df.corr().iloc[0, 1]
        print(f"Pandas computed Pearson r: {overall_pearson_r}")
        mean_overall_pearson_r+=overall_pearson_r
        f, ax = plt.subplots(figsize=(7, 3))
        df.rolling(window=30, center=True).median().plot(ax=ax)
        ax.set(xlabel='Time', ylabel='Pearson r')
        ax.set(title=f"Overall Pearson r = {np.round(overall_pearson_r, 2)}")
        plt.show()
    mean_overall_pearson_r=mean_overall_pearson_r/user_num
    return mean_overall_pearson_r