#Course_Project_Data_Preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import keras
import tensorflow as tf
import keras.backend as K
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

labeldf=newdf['label']
labeldf_test=newdf_test['label']

# change the label column
newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})


# put the new label column back
newdf['label'] = newlabeldf
newdf_test['label'] = newlabeldf_test


to_drop_DoS = [0,1]
to_drop_Probe = [0,2]
to_drop_R2L = [0,3]
to_drop_U2R = [0,4]

#filter coloumns

DoS_df=newdf[newdf['label'].isin(to_drop_DoS)];
Probe_df=newdf[newdf['label'].isin(to_drop_Probe)];
R2L_df=newdf[newdf['label'].isin(to_drop_R2L)];
U2R_df=newdf[newdf['label'].isin(to_drop_U2R)];



#test
DoS_df_test=newdf_test[newdf_test['label'].isin(to_drop_DoS)];
Probe_df_test=newdf_test[newdf_test['label'].isin(to_drop_Probe)];
R2L_df_test=newdf_test[newdf_test['label'].isin(to_drop_R2L)];
U2R_df_test=newdf_test[newdf_test['label'].isin(to_drop_U2R)];


print('Train:')
print('Dimensions of DoS:' ,DoS_df.shape)
print('Dimensions of Probe:' ,Probe_df.shape)
print('Dimensions of R2L:' ,R2L_df.shape)
print('Dimensions of U2R:' ,U2R_df.shape)
print()
print('Test:')
print('Dimensions of DoS:' ,DoS_df_test.shape)
print('Dimensions of Probe:' ,Probe_df_test.shape)
print('Dimensions of R2L:' ,R2L_df_test.shape)
print('Dimensions of U2R:' ,U2R_df_test.shape)


# Split dataframes into X & Y
# X Features , Y result changes

X_DoS = DoS_df.drop('label', axis=1)
Y_DoS = DoS_df['label']

X_Probe = Probe_df.drop('label', axis=1)
Y_Probe = Probe_df['label']

X_R2L = R2L_df.drop('label', axis=1)
Y_R2L = R2L_df['label']

X_U2R = U2R_df.drop('label', axis=1)
Y_U2R = U2R_df['label']

# Test set
X_DoS_test = DoS_df_test.drop('label', axis=1)
Y_DoS_test = DoS_df_test['label']

X_Probe_test = Probe_df_test.drop('label', axis=1)
Y_Probe_test = Probe_df_test['label']

X_R2L_test = R2L_df_test.drop('label', axis=1)
Y_R2L_test = R2L_df_test['label']

X_U2R_test = U2R_df_test.drop('label', axis=1)
Y_U2R_test = U2R_df_test['label']


colNames=list(X_DoS)
colNames_test=list(X_DoS_test)

from sklearn import preprocessing

scaler1 = preprocessing.StandardScaler().fit(X_DoS)
X_DoS=scaler1.transform(X_DoS)

scaler2 = preprocessing.StandardScaler().fit(X_Probe)
X_Probe=scaler2.transform(X_Probe)

scaler3 = preprocessing.StandardScaler().fit(X_R2L)
X_R2L=scaler3.transform(X_R2L)

scaler4 = preprocessing.StandardScaler().fit(X_U2R)
X_U2R=scaler4.transform(X_U2R)

# test data
scaler5 = preprocessing.StandardScaler().fit(X_DoS_test)
X_DoS_test=scaler5.transform(X_DoS_test)

scaler6 = preprocessing.StandardScaler().fit(X_Probe_test)
X_Probe_test=scaler6.transform(X_Probe_test)

scaler7 = preprocessing.StandardScaler().fit(X_R2L_test)
X_R2L_test=scaler7.transform(X_R2L_test)

scaler8 = preprocessing.StandardScaler().fit(X_U2R_test)
X_U2R_test=scaler8.transform(X_U2R_test)




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


y_binary = to_categorical(Y_Df, num_classes=5)
y_test_binary = to_categorical(Y_Df_test, num_classes=5)


#############
RANDOM FOREST
#############

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Create a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_Df, Y_Df)

# Make predictions on the test set
y_pred = rf.predict(X_Df_test)

# Evaluate the model
print("Classification Report:\n", classification_report(Y_Df_test, y_pred))
print("Accuracy:", accuracy_score(Y_Df_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_Df_test, y_pred))

# Define the grid of hyperparameters
grid_space = {
    'max_depth': [3, 5, 10, None],
    'n_estimators': [10, 100, 200],
    'max_features': [1, 3, 5, 7],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 3]  # Avoid setting min_samples_split=1, which can lead to overfitting
}

# Create and fit the GridSearchCV model
grid = GridSearchCV(estimator=rf, param_grid=grid_space, cv=3, scoring='accuracy')
model_grid = grid.fit(X_Df, Y_Df)

# Print the best hyperparameters and best score
print('Best hyperparameters are:', model_grid.best_params_)
print('Best score is:', model_grid.best_score_)

######################
SUPPORT VECTOR MACHINE
######################

from sklearn.svm import SVC

# prompt: Define SVC, build, train and evaluate SVM algorithm, experiment with different kernels

# Create an SVM classifier with a linear kernel
svm_linear = SVC(kernel='linear', random_state=42)

# Train the model
svm_linear.fit(X_Df, Y_Df)

# Make predictions on the test set
y_pred_linear = svm_linear.predict(X_Df_test)

# Evaluate the model
print("Linear Kernel:")
print(classification_report(Y_Df_test, y_pred_linear))
print("Accuracy:", accuracy_score(Y_Df_test, y_pred_linear))
print("Confusion Matrix:\n", confusion_matrix(Y_Df_test, y_pred_linear))

# Create an SVM classifier with a radial basis function (RBF) kernel
svm_rbf = SVC(kernel='rbf', random_state=42)

# Train the model
svm_rbf.fit(X_Df, Y_Df)

# Make predictions on the test set
y_pred_rbf = svm_rbf.predict(X_Df_test)

# Evaluate the model
print("\nRBF Kernel:")
print(classification_report(Y_Df_test, y_pred_rbf))
print("Accuracy:", accuracy_score(Y_Df_test, y_pred_rbf))
print("Confusion Matrix:\n", confusion_matrix(Y_Df_test, y_pred_rbf))

# Create an SVM classifier with a polynomial kernel
svm_poly = SVC(kernel='poly', degree=3, random_state=42)

# Train the model
svm_poly.fit(X_Df, Y_Df)

# Make predictions on the test set
y_pred_poly = svm_poly.predict(X_Df_test)

# Evaluate the model
print("\nPolynomial Kernel:")
print(classification_report(Y_Df_test, y_pred_poly))
print("Accuracy:", accuracy_score(Y_Df_test, y_pred_poly))
print("Confusion Matrix:\n", confusion_matrix(Y_Df_test, y_pred_poly))

##########
DNN
##########

#Create the neural network

dnn = Sequential()
dnn.add(Dense(units=512, activation='relu', input_shape=(X_Df.shape[1],)))
dnn.add(BatchNormalization())
dnn.add(Dropout(0.4))
dnn.add(Dense(units=256, activation='relu'))
dnn.add(BatchNormalization())
dnn.add(Dropout(0.3))
dnn.add(Dense(units=128, activation='relu'))
dnn.add(BatchNormalization())
dnn.add(Dropout(0.3))
dnn.add(Dense(units=5, activation='softmax'))


dnn.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


dnn.fit(X_Df, y_binary, epochs=50, batch_size=128, verbose=1, validation_split=0.1, callbacks=[checkpointer])


loss, results = dnn.evaluate(XDf_test, Y_Df_test, verbose=1)

print(f'Test loss: {loss}')
print(f'Test accuracy: {results}')

#Precision, Recall, FI-Score, and ROC-AUC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, roc_curve

#predict probabiliites for test set
y_prediction = dnn.predict(XDf_test, verbose=0)
y_pred_classes = np.argmax(y_prediction, axis=1)

y_true_classes = Y_Df_test

#accuracy, precision, recall, F1-score
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f'Accuracy: {accuracy *100:.2f}%')

print(classification_report(y_true_classes, y_pred_classes))

#convert y_true_classes to one-hot encoded
y_true_one_hot = to_categorical(y_true_classes, num_classes=5)

#ROC_AUC
try:
    auc = roc_auc_score(y_true_one_hot, y_prediction, multi_class='ovo')
    print(f'ROC AUC: {auc *100:.2f}%', '\n')
except ValueError as e:
    print(f'Error calculating ROC AUC: {e}')



confusion = confusion_matrix(y_true_classes, y_pred_classes)
print('Confusion:')
print(confusion)



import seaborn as sns

axes = sns.heatmap(confusion, annot=True, cmap='nipy_spectral_r')






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
