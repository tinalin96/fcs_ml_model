import pandas as pd
import os
import glob

#read marker and label files
labels = pd.read_excel('/Users/tina/Desktop/EU_label.xlsx')
marker = pd.read_excel('/Users/tina/Desktop/EU_marker_channel_mapping.xlsx')

#scan for .fcs data filepaths
dir_path = '/Users/tina/Desktop/raw_fcs/**/*.fcs'
filepath = [file for file in glob.glob(dir_path, recursive=True)]

#filter channel and read all fcs data
marker_channel=marker[marker.use==1]['marker_channel'].to_list()
raw_data = [FlowCal.io.FCSData(file) for file in filepath]
live_channel_list=list(set(marker_channel) & set(raw_data[4].channels))

#construct dataframe to fit channel statistics (40,60)
feature_df=pd.DataFrame([])
for i in range(40):
    raw_data[i]=raw_data[i][:,live_channel_list]
    new_data={}
    for channel in live_channel_list:
        new_data.update({
            channel+'_cv':FlowCal.stats.cv(raw_data[i], channels=channel),
            channel+'_gcv':FlowCal.stats.gcv(raw_data[i], channels=channel),
            channel+'_gmean':FlowCal.stats.gmean(raw_data[i], channels=channel),
            channel+'_gstd':FlowCal.stats.gstd(raw_data[i], channels=channel),
            channel+'_iqr':FlowCal.stats.iqr(raw_data[i], channels=channel),
            channel+'_mean':FlowCal.stats.mean(raw_data[i], channels=channel),
            channel+'_median':FlowCal.stats.median(raw_data[i], channels=channel),
            channel+'_mode':FlowCal.stats.mode(raw_data[i], channels=channel),
            channel+'_rcv':FlowCal.stats.rcv(raw_data[i], channels=channel),
            channel+'_std':FlowCal.stats.std(raw_data[i], channels=channel)})
    new_data.update({"file":raw_data[i].infile})
    feature_df=feature_df.append(new_data, ignore_index=True)

#sort features to match label
raw_data.sort(key=lambda x: x.infile, reverse=False)
sorted_labels = labels.sort_values(by=['label'], ascending=True).replace(['Healthy','Sick'], [0, 1])
sorted_features = feature_df.sort_values(by=['file'], ascending=True).drop(columns=['file'])

#split train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sorted_features, sorted_labels['label'], test_size=0.3,random_state=109) # 70% training and 30% test

#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='poly') # Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


