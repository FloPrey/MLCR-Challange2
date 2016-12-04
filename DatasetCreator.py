import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from random import randint

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
                          'Participant_ID': test_df.subject.iloc[0],
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

def computeBasicFeatures(raw_df):
    """Method to invoke the feature creation method."""

    feature_df = pd.DataFrame()  # not declared in sample code
    for subject_id, subject_df in raw_df.groupby(raw_df.subject):
        for test_id, test_df in subject_df.groupby(subject_df.test):
            feature_df = feature_df.append(compute_features(test_df), ignore_index=True)
    feature_df.reset_index(inplace=True, drop=True)

    # Compute the time of day as a float
    h = feature_df.Test_time.apply(lambda x: x.hour)
    m = feature_df.Test_time.apply(lambda x: x.minute)
    feature_df['time_as_float'] = h + (m / 60.0)

    return feature_df

def joinFeaturesAndDiary(feature_df, diary_df):
    """Method to join the diary information (alcohol, food consumption...)
    with the recorded reaction time data."""

    newTimeList = list()
    timeList = feature_df["Test_time"].tolist()

    for item in timeList:

        # round the minutes of the time of the reaction test to fit the 15 min steps of the diary
        rounded_qtr_hour = 15*(item.minute // 15)

        newTimeList.append(str(item.replace(minute=rounded_qtr_hour, second=0)))

    feature_df["Time"] = newTimeList

    inputData = pd.merge(feature_df, diary_df, on=['Participant_ID','Time'])

    return inputData

def extender(diary, label, expand):
    """
    There are some columns in the dataset which describe feature, which may have a lasting impact.
    This method tries to enhance the dataset to store the impact. This is achieved to set the columns after the event
    to 1.
    :param diary: the dataset to prepare
    :param label: the columns to mod. May be list[str] or str
    :param expand: The factor to expand. If 0, the dataset remains the same. If n, the n fields after the event are set
    to true.
    :return:
    """
    if isinstance(label, list):
        for e in label:
            extender(diary, e, expand)
    else:
        ref = diary[label].tolist()
        mod = list(diary[label].tolist())

        for i in range(0, len(ref)):
            if ref[i] == 1:
                for y in range(0, expand):
                    y = i + y + 1
                    if y >= len(ref):
                        y = len(ref) - 1
                    mod[y] = 1

        diary[label] = mod
        
def createDataSet():
    '''Creates the complete Dataset containing all features, diary-labels and output label.'''
    
    # load rawData
    raw_df = pd.read_hdf("dataset/data.h5", "raw")
    # load enhanced diary
    diary_df = pd.read_csv('dataset/modifiedLabels.csv')
    # enhance diary
    extender(diary=diary_df, label=["Alcohol", "Caffeine", "Food", "Medication", "Nicotine", "Sports"], expand=3)
    # change participant numbers as there are only 7
    raw_df.loc[raw_df['subject'] == 7, 'subject'] = 6
    raw_df.loc[raw_df['subject'] == 8, 'subject'] = 7

    # compute basicFeatures
    feature_df = computeBasicFeatures(raw_df)

    # join feature set with diary entries (needs Participant_ID and timestamp to join)
    inputDataSet = joinFeaturesAndDiary(feature_df, diary_df)

    # add the output labels (msf) to the dataset
    completeDataset_df = joinOutputLabel(inputDataSet)

    completeDataset_df.to_csv("dataset/completeDataset")

    return completeDataset_df

def inputOutputDataWorkDays(completeDataset_df):
    """Method returns all data taken / recorded on a workdays."""

    numberOfParticipants = int(completeDataset_df["Participant_ID"].max())

    features = ['Participant_ID', 'day', 'msf', 'MSFSC', 'Alarmclock', "Alcohol", "Caffeine", "Food", "Medication", "Nicotine", "Sports", 'earliest_test', 'earliest_time', 'best_mean', 'time_of_best', 'latest_test', 'latest_time']

    inputOutput_df = pd.DataFrame(columns = features)

    for participant in range(numberOfParticipants):

        participant = participant+1

        # get all all data of current participant
        participantData = completeDataset_df.loc[(completeDataset_df.Participant_ID == participant)]

        for day in range(participantData["day"].max()):

            day = day+1

            if (any(participantData.day == day)):

                # get all data of current participant at the current awake cycle
                # only select if the day is a workday
                participantsDayData = participantData.loc[(participantData.day == day) & (participantData.Workday == 1)]

                if (len(participantsDayData)>0):
                    
                    # set containing main features, labels and the earliest test of the awake cycle
                    earliestTest = participantsDayData.loc[(participantsDayData['Test_nr'].idxmin()), ['Participant_ID', 'day', 'msf', 'MSFSC', 'Alarmclock', "Alcohol", "Caffeine", "Food", "Medication", "Nicotine", "Sports", 'positive_mean', 'time_as_float']].tolist()
                    
                    # set containing the best test of the current awake cycle
                    bestMeanOfDay = participantsDayData.loc[(participantsDayData['positive_mean'].idxmin()), ['positive_mean', 'time_as_float']].tolist()

                    # set containg the latest test of the awake cycle
                    latestTest = participantsDayData.loc[(participantsDayData['Test_nr'].idxmax()), ['positive_mean', 'time_as_float']].tolist()

                    earliestTest.extend(bestMeanOfDay)
                    earliestTest.extend(latestTest)
                    
                    # append all feauteres to the dataframe
                    inputOutput_df.loc[len(inputOutput_df)]=earliestTest

    return inputOutput_df

def inputOutputDataFreeDays(completeDataset_df):
    """Method returns all data taken / recorded on a freedays."""

    numberOfParticipants = int(completeDataset_df["Participant_ID"].max())

    features = ['Participant_ID', 'day', 'msf', 'MSFSC', 'Alarmclock', "Alcohol", "Caffeine", "Food", "Medication", "Nicotine", "Sports", 'earliest_test', 'earliest_time', 'best_mean', 'time_of_best', 'latest_test', 'latest_time']

    inputOutput_df = pd.DataFrame(columns = features)

    for participant in range(numberOfParticipants):

        participant = participant+1

        # get all all data of current participant
        participantData = completeDataset_df.loc[(completeDataset_df.Participant_ID == participant)]

        for day in range(participantData["day"].max()):

            day = day+1

            if (any(participantData.day == day)):
                
                # get all data of current participant at the current awake cycle
                # only select if the day is not a workday
                participantsDayData = participantData.loc[(participantData.day == day) & (participantData.Workday == 0)]

                if (len(participantsDayData)>0):
                    
                    # set containing main features, labels and the earliest test of the awake cycle
                    earliestTest = participantsDayData.loc[(participantsDayData['Test_nr'].idxmin()), ['Participant_ID', 'day', 'msf', 'MSFSC', 'Alarmclock', "Alcohol", "Caffeine", "Food", "Medication", "Nicotine", "Sports", 'positive_mean', 'time_as_float']].tolist()
                    
                    # set containing the best test of the current awake cycle
                    bestMeanOfDay = participantsDayData.loc[(participantsDayData['positive_mean'].idxmin()), ['positive_mean', 'time_as_float']].tolist()
                    
                    # set containg the latest test of the awake cycle
                    latestTest = participantsDayData.loc[(participantsDayData['Test_nr'].idxmax()), ['positive_mean', 'time_as_float']].tolist()

                    earliestTest.extend(bestMeanOfDay)
                    earliestTest.extend(latestTest)

                    # append all feauteres to the dataframe
                    inputOutput_df.loc[len(inputOutput_df)]=earliestTest

    return inputOutput_df

def splitLabels(inputOutput_df):
    """Method to split labels from the input dataset."""

    outputData = inputOutput_df["msf"].tolist()
    inputData = inputOutput_df.copy()
    inputData.drop('msf', axis=1, inplace=True)

    return inputData, outputData

def splitDataset(inputOutput_df):
    """Method to split the dataset into train and test sets and their
    respective outputs. The testsets contains one random day of each participant
    if possible."""

    numberOfParticipants = int(inputOutput_df["Participant_ID"].max())

    train_x = inputOutput_df.copy()

    test_x = pd.DataFrame(columns = list(inputOutput_df.columns.values))

    for participant in range(numberOfParticipants):

        participant = participant+1

        # get all all data of current participant
        participantData = train_x.loc[(train_x.Participant_ID == participant)]

        if (len(participantData) > 1):

            testDataIndex = randint(0, len(participantData)-1)

            test_x.loc[len(test_x)]=participantData.iloc[testDataIndex]

            train_x.drop(train_x.index[testDataIndex])

    # extract and delete the output values
    train_y = train_x["msf"].tolist()
    train_x.drop('msf', axis=1, inplace=True)
    
    # extract and delete the output values
    test_y = test_x["msf"].tolist()
    test_x.drop('msf', axis=1, inplace=True)

    return train_x, train_y, test_x, test_y

'''Helper method to calculate the msf values for the entire dataset.'''
def createMSFMatrix():
    """Matrix to compute all mid-sleep data. """

    # load already enhanced label-set (added day column)
    diary = pd.read_csv('dataset/modifiedLabels.csv')

    numberOfParticipants = diary["Participant_ID"].max()
    numberOfDays = len(set(diary["day"].tolist()))

    # create matrix containing the participants and their mid-sleep time for each day
    msfMatrix = np.zeros([numberOfParticipants, numberOfDays], dtype=float)

    for participant in range(numberOfParticipants):

        participant = participant+1

        participantSleepData = diary[(diary.Sleep == 1) & (diary.Participant_ID == participant)]

        for day in range((participantSleepData["day"].max())):

            day = day+1
            participantDayData = participantSleepData[participantSleepData.day == day]
            
            # sleep duration from diary data - every row corresponds to 15 minutes
            sleepDuration = len(participantDayData) * 15
            sleepOnset = participantDayData["Time"].iloc[0]

            sleepOnset = datetime.strptime(sleepOnset, '%Y-%m-%d %H:%M:%S')
            msf = sleepOnset + timedelta(minutes=(sleepDuration/2))

            hour = msf.hour
            minute = msf.minute

            msfFloat = float(hour + (minute / 60.0))
            msfMatrix[participant-1][day-1] = msfFloat

    return msfMatrix


def median_from_timedelta_tuplelist(raw):
    """
    Calculates Average for MSFsc. Uses median as Average.
    :type raw: list[(datetime, timedelta)]
    :param raw: data for MSFsc
    :return: (sd, so)
    """
    if len(raw) == 0:
        return timedelta(0), timedelta(0)
    m = int((len(raw) - 1) / 2)
    sd = sorted([z for (y, z) in raw])[m]
    so = sorted([y for (y, z) in raw])[m]  # type: datetime
    so = timedelta(hours=so.hour, minutes=so.minute)
    return sd, so


def avg_from_timedelta_tuplelist(raw):
    """
    Calculates Average for MSFsc. Uses arithmetical middle as Average.
    :type raw: list[(datetime, timedelta)]
    :param raw: data for MSFsc
    :return: (sd, so)
    """
    if len(raw) == 0:
        return timedelta(0), timedelta(0)
    sd = sum([z for (y, z) in raw], timedelta(0)) / len(raw)

    so = datetime.fromtimestamp(sum([y.timestamp() for (y, z) in raw]) / len(raw))
    so = timedelta(hours=so.hour, minutes=so.minute)
    return sd, so


def createMSF_SCMatrix(middle=median_from_timedelta_tuplelist):
    """
    Method to create MSF_SC for participants. Estimate, that MSF_SC is calculated over whole test area.
    :param middle: MSF_SC is a combined value. It consists out of many times. This function is used to pick the middle
    value.
    :return: Dataframe with index=participant_id and column MSFSC
    """
    # load already enhanced label-set (added day column)
    diary = pd.read_csv('dataset/modifiedLabels.csv')
    numberOfParticipants = diary["Participant_ID"].max()
    msf_persist = {"Participant_ID": [], "MSFSC": []}
    for participant in range(numberOfParticipants):
        participant = participant + 1
        # Get all all sleep moments of the current participant
        participantSleepData = diary[(diary.Sleep == 1) & (diary.Participant_ID == participant)]
        # split data for work / free days
        raw_work = []
        raw_free = []
        for day in range((participantSleepData["day"].max())):
            # collect data and prepare for avg calculation
            day = day + 1
            participantDayData = participantSleepData[participantSleepData.day == day]
            sleepOnset = participantDayData["Time"].iloc[0]
            sleepOnset = datetime.strptime(sleepOnset, '%Y-%m-%d %H:%M:%S')
            # normalize so for later average calculation
            sleepOnset = sleepOnset - timedelta(days=day)
            sleepDuration = timedelta(minutes=(len(participantDayData) * 15))
            # onset, duration
            tuple = (sleepOnset, sleepDuration)
            if participantDayData["Workday"].values[0] == 1:
                raw_work.append(tuple)
            else:
                raw_free.append(tuple)
        # avg calculation
        # calculate data for msfsc
        sd_f, so_f = middle(raw_free)
        sd_w, so_w = middle(raw_work)
        sd_week = (sd_f * len(raw_free) + sd_w * len(raw_work)) / (len(raw_free) + len(raw_work))
        msf = so_f + (sd_f / 2)
        msf_sc = msf
        if (sd_f > sd_w):
            # print("special case: sd_f > sd_w")
            msf_sc = msf - ((sd_f - sd_week) / 2)  # type: timedelta
        msf_sc = msf_sc.seconds / float(3600)
        msf_persist["Participant_ID"].append(participant)
        msf_persist["MSFSC"].append(msf_sc)

    index = msf_persist["Participant_ID"]
    del (msf_persist["Participant_ID"])
    # msfsc for each participant
    msfMatrix = pd.DataFrame(msf_persist, index=index)
    return msfMatrix

def joinOutputLabel(inputData):
    """
    Joins the input dataset with MSF(label) and MSFsc
    :param inputData: the input dataset.
    :return: the new dataset with msf and msfsc
    """
    msfMatrix = createMSFMatrix()
    inputAndOutput_df = inputData.copy()
    msfColumn = list()
    # add msf
    for i in range(0, len(inputData)):
        msfColumn.append(msfMatrix[int(inputData.iloc[i]['Participant_ID']) - 1][int(inputData.iloc[i]['day']) - 1])

    inputAndOutput_df['msf'] = msfColumn
    # add msfsc
    msfsc = createMSF_SCMatrix(middle=median_from_timedelta_tuplelist)
    inputAndOutput_df = inputAndOutput_df.join(msfsc, on='Participant_ID')
    return inputAndOutput_df

def splitByParticipant(train_x, train_y, test_x, test_y):
    amountIds = len(set(train_x['Participant_ID']))
    totalIds = len(train_x['Participant_ID'])
    participantTrainFeatures = []
    participantTrainTargets = []
    participantTestFeatures = []
    participantTestTargets = []
    startRow = 0
    endRow = 0
    for i in range(amountIds):
        currentID = train_x['Participant_ID'][startRow]
        while (endRow<totalIds) and (currentID == train_x['Participant_ID'][endRow]):
            endRow = endRow + 1

        participantTrainFeatures.append(train_x[:][startRow:endRow])
        participantTestFeatures.append(test_x[:][startRow:endRow])
        participantTrainTargets.append(train_y[:][startRow:endRow])
        participantTestTargets.append(test_y[:][startRow:endRow])
        startRow = endRow

    return participantTrainFeatures, participantTrainTargets, participantTestFeatures, participantTestTargets
