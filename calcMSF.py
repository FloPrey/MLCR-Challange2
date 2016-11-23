import pandas as pd
from sklearn.cluster import KMeans
df_data = pd.read_hdf('dataset/data.h5', 'labels')

sleep_freetime_data = df_data[(df_data.Sleep == 1) & (df_data.Workday == 0) & (df_data.Participant_ID == 1)]
sleep_weekday_data = df_data[(df_data.Sleep == 1) & (df_data.Workday == 1) & (df_data.Participant_ID == 1)]
anz_weekday_data = len(df_data[(df_data.Workday == 1) & (df_data.Participant_ID == 1)])/24/4
anz_freeday_data = len(df_data[(df_data.Workday == 0) & (df_data.Participant_ID == 1)])/24/4
sleep_min_freetime = len(sleep_freetime_data) * 15
sleep_min_week = len(sleep_weekday_data) * 15


msf = sleep_min_freetime/2
sd_week = (sleep_min_week*anz_weekday_data + sleep_min_freetime*anz_freeday_data) / 7
if sleep_min_freetime > sleep_min_week:
    msf = msf - (sleep_min_freetime-sd_week)/2

msf_hour = msf / 60
msf_min = msf % msf_hour
print(str(msf_hour)+":"+str(msf_min))


"""for row in sleep_freetime_data.iterrows():
    if row[6] in sleep_per_participant:
        sleep_per_participant[row[6]] += 15
    else:
        sleep_per_participant.insert(row[6], 15)

print(sleep_per_participant)"""