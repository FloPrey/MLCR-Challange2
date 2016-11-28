import pandas as pd
import numpy as np
from datetime import datetime, timedelta    

def get_quantile(data, q):
    """Takes series of values and returns quantile limit as well as the mean of the values above the quantile.
    data: Data as pandas Series.
    q: Quantile (0.75 -> 75%)
    returns: quantile limit, mean value of elements above quantile limit
    """

    quantile_limit = data.quantile(q=q)
    quantile_mean = data[data >= quantile_limit].mean()
    return quantile_limit, quantile_mean


def compute_features(test_df, verbose=False):
    """ Takes PVT test results and returns feature vector as a result.
    test_df: Dataframe containing PVT test results.
    Returns: Series containing the feature vector.
    """
    test_time = test_df.timestamp.iloc[0]
    n = test_df.shape[0]
    positive_data = test_df[test_df.response_time > 0]  # drop all "too early samples"
    n_positive = positive_data.shape[0]
    positive_mean = positive_data.response_time.mean()
    positive_median = positive_data.response_time.median()
    positive_std = positive_data.response_time.std()
    q50_lim, q50_mean = get_quantile(positive_data.response_time, 0.50)
    q75_lim, q75_mean = get_quantile(positive_data.response_time, 0.75)
    q90_lim, q90_mean = get_quantile(positive_data.response_time, 0.90)
    q95_lim, q95_mean = get_quantile(positive_data.response_time, 0.95)
    features = pd.Series({'Test_time': test_time,
                          'Subject': test_df.subject.iloc[0],
                          'Test_nr': test_df.test.iloc[0],
                          'n_total': n,
                          'n_positive': n_positive,
                          'positive_mean': positive_mean,
                          'positive_median': positive_median,
                          'positive_std': positive_std,
                          'q50_lim': q50_lim,
                          'q75_lim': q75_lim,
                          'q90_lim': q90_lim,
                          'q95_lim': q95_lim,
                          'q50_mean': q50_mean,
                          'q75_mean': q75_mean,
                          'q90_mean': q90_mean,
                          'q95_mean': q95_mean})
    if verbose:
        print(features)
    return features


def createInputData():
    
    raw_df = pd.read_hdf("dataset/data.h5", "raw")
    
    # change participant numbers as there are only 7
    raw_df.loc[raw_df['subject'] == 7, 'subject'] = 6
    raw_df.loc[raw_df['subject'] == 8, 'subject'] = 7
    
    feature_df = pd.DataFrame()  # not declared in sample code
    for subject_id, subject_df in raw_df.groupby(raw_df.subject):
        for test_id, test_df in subject_df.groupby(subject_df.test):
            feature_df = feature_df.append(compute_features(test_df), ignore_index=True)
    feature_df.reset_index(inplace=True, drop=True)
    
    
    # Compute the time of day as a float
    h = feature_df.Test_time.apply(lambda x: x.hour)
    m = feature_df.Test_time.apply(lambda x: x.minute)
    feature_df['time_of_day'] = h + (m / 60.0)
    
    return selectFeature(feature_df)

'''Add further feature selection'''
def selectFeature(feature_df):
    
    subFeatureSet = feature_df.loc[(feature_df.Subject == 1), ['positive_mean', 'time_of_day']]
    
    return subFeatureSet
    
def createOutputData():
    
    # load already enhanced label-set (added day column)
    diary = pd.read_csv('dataset/modifiedLabels.csv')

    numberOfParticipants = diary["Participant_ID"].max()
    numberOfDays = len(set(diary["day"].tolist()))
    
    # create matrix containing the participants and their msf for each day
    msfMatrix = np.zeros([numberOfParticipants, numberOfDays], dtype=float)
    
    for participant in range(numberOfParticipants):
        
        participant = participant+1
        
        # Get all all sleep moments of the current participant
        participantSleepData = diary[(diary.Sleep == 1) & (diary.Participant_ID == participant)]
        
        for day in range((participantSleepData["day"].max())):
               
            day = day+1     
            participantDayData = participantSleepData[participantSleepData.day == day]
            
            sleepDuration = len(participantDayData) * 15        
            sleepOnset = participantDayData["Time"].iloc[0]
            
            sleepOnset = datetime.strptime(sleepOnset, '%Y-%m-%d %H:%M:%S') 
            msf = sleepOnset + timedelta(minutes=(sleepDuration/2))
            
            hour = msf.hour
            minute = msf.minute
            
            msfFloat = float(hour + (minute / 60.0))
            msfMatrix[participant-1][day-1] = msfFloat
    
    return msfMatrix