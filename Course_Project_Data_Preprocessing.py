#Course_Project_Data_Preprocessing

import pandas as pd
import numpy as np
import sys
import sklearn
import io
import random

#Load the dataset

kdd_Train = pd.read_csv('KDDTrain.csv')
kdd_Test = pd.read_csv('KDDTest.csv')

train_path = r'C:\Users\wtene\OneDrive\Documents\USM\CCS691\KDDTrain.csv'
test_path = r'C:\Users\wtene\OneDrive\Documents\USM\CCS691\KDDTest.csv'


column_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']

train = pd.read_csv(train_path, header=None, names = column_names)
test = pd.read_csv(test_path, header=None, names = column_names)

print('Training set: ', train.shape)
print('Test set: ', test.shape)


#Explore dataset
print('Label Count for Training Set: ')
print(train['label'].value_counts())
print()
print('Label Count for Test Set: ')
print(test ['label'].value_counts())

print('Training set:')
for column_names in train.columns:
    if train[column_names].dtypes == 'object' :
        unique_cat = len(train[column_names].unique())
        print("Feature '{column_names}' has {unique_cat} categories".format(column_names=column_names, unique_cat=unique_cat))

print()
print('Label distribution: ')
print(train['service'].value_counts().sort_values(ascending=False).head())

print('Test set:')
for column_names in test.columns:
    if test[column_names].dtypes == 'object' :
        unique_cat = len(test[column_names].unique())
        print("Feature '{column_names}' has {unique_cat} categories".format(column_names=column_names, unique_cat=unique_cat))


#One Hot Encoding

import numpy as np

print('Protocol Type: ', train['protocol_type'].unique())
print('Service: ', train['service'].unique())
print('Flag: ', train['flag'].unique())

print('Protocol Type Value Count', train['protocol_type'].value_counts())
print('Service Value Count', train['service'].value_counts())
print('Flag Value Count', train['protocol_type'].value_counts())

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
categorical_columns=['protocol_type', 'service', 'flag']

train_categorical_values = train[categorical_columns]
test_categorical_values = test[categorical_columns]

train_categorical_values.head()



# protocol type
unique_protocol=sorted(train.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2=[string1 + x for x in unique_protocol]
print(unique_protocol2)

# service
unique_service=sorted(train.service.unique())
string2 = 'service_'
unique_service2=[string2 + x for x in unique_service]
print(unique_service2)


# flag
unique_flag=sorted(train.flag.unique())
string3 = 'flag_'
unique_flag2=[string3 + x for x in unique_flag]
print(unique_flag2)


# put together
dumcols=unique_protocol2 + unique_service2 + unique_flag2


#for test set
unique_service_test=sorted(test.service.unique())
unique_service2_test=[string2 + x for x in unique_service_test]
testdumcols=unique_protocol2 + unique_service2_test + unique_flag2

train_categorical_values_enc=train_categorical_values.apply(LabelEncoder().fit_transform)
print(train_categorical_values.head())
print('\n')
print(train_categorical_values_enc.head())

# test set
test_categorical_values_enc=test_categorical_values.apply(LabelEncoder().fit_transform)

encoder = OneHotEncoder(categories='auto')
train_categorical_values_encenc = encoder.fit_transform(train_categorical_values_enc)
train_cat_data = pd.DataFrame(train_categorical_values_encenc.toarray(),columns=dumcols)


# test set
test_categorical_values_encenc = encoder.fit_transform(test_categorical_values_enc)
test_cat_data = pd.DataFrame(test_categorical_values_encenc.toarray(),columns=testdumcols)

test_cat_data.head()
     

#To account for missing values

trainservice = train['service'].tolist()
testservice = test['service'].tolist()
difference = list(set(trainservice) - set(testservice))
string = 'service_'
difference = [string + x for x in difference]
difference


for col in difference:
    test_cat_data[col] = 0

print(train_cat_data.shape)    
print(test_cat_data.shape)
     





### SECTION BELOW IS COMMENTED OUT IN CASE NOT NEEDED


'''
newTrain=train.join(train_cat_data)
newTrain.drop('flag', axis=1, inplace=True)
newTrain.drop('protocol_type', axis=1, inplace=True)
newTrain.drop('service', axis=1, inplace=True)

# test data
newTest=test.join(test_cat_data)
newTest.drop('flag', axis=1, inplace=True)
newTest.drop('protocol_type', axis=1, inplace=True)
newTest.drop('service', axis=1, inplace=True)

print(newTrain.shape)
print(newTest.shape)

labelTrain=newTrain['label']
labelTest=newTest['label']
pd.set_option('future.no_silent_downcasting', True)


# change the label column
newlabelTrain=labelTrain.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newlabelTest=labelTest.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})



# put the new label column back
newTrain['label'] = newlabelTrain
newTest['label'] = newlabelTest


to_drop_DoS = [0,1]
to_drop_Probe = [0,2]
to_drop_R2L = [0,3]
to_drop_U2R = [0,4]


DoS_Train=newTrain[newTrain['label'].isin(to_drop_DoS)];
Probe_Train=newTrain[newTrain['label'].isin(to_drop_Probe)];
R2L_Train=newTrain[newTrain['label'].isin(to_drop_R2L)];
U2R_Train=newTrain[newTrain['label'].isin(to_drop_U2R)];



#test
DoS_Test=newTest[newTest['label'].isin(to_drop_DoS)];
Probe_Test=newTest[newTest['label'].isin(to_drop_Probe)];
R2L_Test=newTest[newTest['label'].isin(to_drop_R2L)];
U2R_Test=newTest[newTest['label'].isin(to_drop_U2R)];


print('Train:')
print('Dimensions of DoS:' ,DoS_Train.shape)
print('Dimensions of Probe:' ,Probe_Train.shape)
print('Dimensions of R2L:' ,R2L_Train.shape)
print('Dimensions of U2R:' ,U2R_Train.shape)
print()
print('Test:')
print('Dimensions of DoS:' ,DoS_Test.shape)
print('Dimensions of Probe:' ,Probe_Test.shape)
print('Dimensions of R2L:' ,R2L_Test.shape)
print('Dimensions of U2R:' ,U2R_Test.shape)
'''
