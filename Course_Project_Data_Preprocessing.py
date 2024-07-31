#Course_Project_Data_Preprocessing

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import preprocessing
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

#Add the columns of the dataset

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

#Load the datasets
df = pd.read_csv("KDDTrain.csv", header=None, names = column_names)
df_test = pd.read_csv("KDDTest.csv", header=None, names = column_names)

print('Dimensions of the Training set: ', df.shape)
print('Dimensions of the Test set: ', df_test.shape)

df.head(5)

#Explore dataset
print('Label Count for Training Set: ')
print(df['label'].value_counts())
print()
print('Label Count for Test Set: ')
print(df_test['label'].value_counts())

#columns are categorical, not yet binary

print('Training set:')
for column_names in df.columns:
    if df[column_names].dtypes == 'object' :
        unique_cat = len(df[column_names].unique())
        print("Feature '{column_names}' has {unique_cat} categories".format(column_names=column_names, unique_cat=unique_cat))

print()
print('Label distribution: ')
print(df['service'].value_counts().sort_values(ascending=False).head())



#Test set
print('Test set:')
for column_names in df_test.columns:
    if df_test[column_names].dtypes == 'object' :
        unique_cat = len(df_test[column_names].unique())
        print("Feature '{column_names}' has {unique_cat} categories".format(column_names=column_names, unique_cat=unique_cat))

print()
print('Label distribution: ')
print(df_test['service'].value_counts().sort_values(ascending=False).head())

#LabelEncoder Insert categorical features into a 2D numpy array

categorical_columns=['protocol_type', 'service', 'flag']

df_categorical_values = df[categorical_columns]
testdf_categorical_values = df_test[categorical_columns]

df_categorical_values.head()


#Had to keep for One-Hot Encoding

# protocol type
unique_protocol=sorted(df.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2=[string1 + x for x in unique_protocol]
#print(unique_protocol2)

# service
unique_service=sorted(df.service.unique())
string2 = 'service_'
unique_service2=[string2 + x for x in unique_service]
#print(unique_service2)


# flag
unique_flag=sorted(df.flag.unique())
string3 = 'flag_'
unique_flag2=[string3 + x for x in unique_flag]
#print(unique_flag2)


# put together
dumcols=unique_protocol2 + unique_service2 + unique_flag2


#do it for test set
unique_service_test=sorted(df_test.service.unique())
unique_service2_test=[string2 + x for x in unique_service_test]
testdumcols=unique_protocol2 + unique_service2_test + unique_flag2


#Transform categorical features into numberes using LabelEncoder()
df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)

print(df_categorical_values.head())
print('--------------------')
print(df_categorical_values_enc.head())

# test set
testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)


#One-Hot Encoding

encoder = OneHotEncoder(categories='auto')
df_categorical_values_encenc = encoder.fit_transform(df_categorical_values_enc)
df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)


# test set
testdf_categorical_values_encenc = encoder.fit_transform(testdf_categorical_values_enc)
testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdumcols)

df_cat_data.head()
     
     

#Missing columns in the test set are added

trainservice = df['service'].tolist()
testservice = df_test['service'].tolist()
difference = list(set(trainservice) - set(testservice))
string = 'service_'
difference = [string + x for x in difference]
difference



for col in difference:
    testdf_cat_data[col] = 0

print(df_cat_data.shape)    
print(testdf_cat_data.shape)



####Could not remove label changes and get code to work because of standardization and scaler

#Add additional columns

newdf=df.join(df_cat_data)
newdf.drop('flag', axis=1, inplace=True)
newdf.drop('protocol_type', axis=1, inplace=True)
newdf.drop('service', axis=1, inplace=True)

# test data
newdf_test=df_test.join(testdf_cat_data)
newdf_test.drop('flag', axis=1, inplace=True)
newdf_test.drop('protocol_type', axis=1, inplace=True)
newdf_test.drop('service', axis=1, inplace=True)

print(newdf.shape)
print(newdf_test.shape)





# change the label column


labelTrain=newTrain['label']
labelTest=newTest['label']

pd.set_option('future.no_silent_downcasting', True)

# change the label column
newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1,
                            'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1,
                              'worm': 1, 'ipsweep' : 1,'nmap' : 1,'portsweep' : 1,'satan' : 1,'mscan' : 1,
                              'saint' : 1, 'ftp_write': 1,'guess_passwd': 1,'imap': 1,'multihop': 1,'phf': 1,
                              'spy': 1,'warezclient': 1,'warezmaster': 1,'sendmail': 1,'named': 1,'snmpgetattack': 1,
                              'snmpguess': 1,'xlock': 1,'xsnoop': 1,'httptunnel': 1,'buffer_overflow': 1,'loadmodule': 1,
                              'perl': 1,'rootkit': 1,'ps': 1,'sqlattack': 1,'xterm': 1 })
newlabeldf_test=labeldf_test.replace({'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
                                        'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,'ipsweep' : 1,
                                        'nmap' : 1,'portsweep' : 1,'satan' : 1,'mscan' : 1,'saint' : 1,'ftp_write': 1,'guess_passwd': 1,
                                        'imap': 1,'multihop': 1,'phf': 1,'spy': 1,'warezclient': 1,'warezmaster': 1,'sendmail': 1,
                                        'named': 1,'snmpgetattack': 1,'snmpguess': 1,'xlock': 1,'xsnoop': 1,'httptunnel': 1,
                                        'buffer_overflow': 1,'loadmodule': 1,'perl': 1,'rootkit': 1,'ps': 1,'sqlattack': 1,'xterm': 1})

# put the new label column back
newdf['label'] = newlabeldf
newdf_test['label'] = newlabeldf_test

x = newdf.drop(['duration','land','wrong_fragment','urgent','num_failed_logins','logged_in','num_compromised',
                'num_file_creations','num_root','root_shell','su_attempted','num_shells','num_access_files',
                'num_outbound_cmds','is_host_login','is_guest_login','flag_S2','flag_S3','flag_SH','srv_rerror_rate',
                'service_csnet_ns','service_ctf','service_daytime','service_discard','service_domain','service_domain_u',
                'service_echo','service_eco_i','service_ecr_i','service_efs','service_exec','service_finger','service_ftp',
                'service_ftp_data','service_gopher','service_netbios_ns','service_ldap','service_kshell','service_klogin',
                'service_iso_tsap','service_imap4','service_http_443','service_hostnames','service_netbios_dgm','service_name',
                'service_mtp','service_login','service_link','service_pop_3','service_pop_2','service_pm_dump','service_other',
                'service_ntp_u','service_nntp','service_nnsp','service_netstat','service_netbios_ssn','service_ssh',
                'service_sql_net','service_sunrpc','service_smtp','service_shell','service_rje','service_remote_job',
                'service_private','service_printer','service_uucp_path','service_uucp','service_urp_i','service_time',
                'service_tim_i','service_tftp_u','service_telnet','service_systat','service_supdup','dst_host_count',
                'srv_diff_host_rate','diff_srv_rate','flag_S0','flag_S1','rerror_rate','flag_RSTR','flag_RSTOS0','flag_RSTO',
                'flag_REJ','flag_OTH','service_whois','service_vmnet','srv_serror_rate','serror_rate','service_urh_i',
                'service_red_i','service_harvest','service_http_2784','dst_host_srv_rerror_rate','dst_host_rerror_rate',
                'dst_host_srv_serror_rate','dst_host_serror_rate','dst_host_srv_diff_host_rate','Protocol_type_tcp',
                'Protocol_type_udp','service_IRC','service_X11','service_Z39_50','service_auth','service_bgp','service_courier',
                'service_http_8001','service_aol'], axis=1)

x_test =newdf_test.drop(['duration','land','wrong_fragment','urgent','num_failed_logins','logged_in','num_compromised',
                         'num_file_creations','num_root','root_shell','su_attempted','num_shells','num_access_files',
                         'num_outbound_cmds','is_host_login','is_guest_login','flag_S2','flag_S3','flag_SH','srv_rerror_rate',
                         'service_csnet_ns','service_ctf','service_daytime','service_discard','service_domain','service_domain_u',
                         'service_echo','service_eco_i','service_ecr_i','service_efs','service_exec','service_finger','service_ftp',
                         'service_ftp_data','service_gopher','service_netbios_ns','service_ldap','service_kshell','service_klogin',
                         'service_iso_tsap','service_imap4','service_http_443','service_hostnames','service_netbios_dgm','service_name',
                         'service_mtp','service_login','service_link','service_pop_3','service_pop_2','service_pm_dump','service_other',
                         'service_ntp_u','service_nntp','service_nnsp','service_netstat','service_netbios_ssn','service_ssh',
                         'service_sql_net','service_sunrpc','service_smtp','service_shell','service_rje','service_remote_job',
                         'service_private','service_printer','service_uucp_path','service_uucp','service_urp_i','service_time',
                         'service_tim_i','service_tftp_u','service_telnet','service_systat','service_supdup','dst_host_count',
                         'srv_diff_host_rate','diff_srv_rate','flag_S0','flag_S1','rerror_rate','flag_RSTR','flag_RSTOS0','flag_RSTO',
                         'flag_REJ','flag_OTH','service_whois','service_vmnet','srv_serror_rate','serror_rate','service_urh_i',
                         'service_red_i','service_harvest','service_http_2784','dst_host_srv_rerror_rate','dst_host_rerror_rate',
                         'dst_host_srv_serror_rate','dst_host_serror_rate','dst_host_srv_diff_host_rate','Protocol_type_tcp',
                         'Protocol_type_udp','service_IRC','service_X11','service_Z39_50','service_auth','service_bgp','service_courier',
                         'service_http_8001','service_aol'], axis=1)



#standardization

# Split dataframes into X & Y

X_Df = x.drop('label', axis=1)
Y_Df = newdf.label

# test set
XDf_test = x_test.drop('label', axis=1) #Had to change variable name. It was picking up shape of X_Df. Not sure how.
Y_Df_test = newdf_test.label

X_Df.shape



#columns saved for later use

colNames = list(X_Df)
colNames_test = list(XDf_test)

scaler1 = preprocessing.StandardScaler().fit(X_Df)
X_Df=scaler1.transform(X_Df)

#test data
scaler2 = preprocessing.StandardScaler().fit(XDf_test)
X_Df_test=scaler2.transform(XDf_test)

XDf_test = XDf_test.astype('float32') #corrected from X_Df_test = X_Df.astype nevermind. Had to change variable name to XDF_test due to mishap with X_Df
Y_Df_test= Y_Df_test.astype('float32')


y_binary = to_categorical(Y_Df)
y_test_binary = to_categorical(Y_Df_test)







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
