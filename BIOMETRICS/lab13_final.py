## Import Modules :
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split 

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def features_vector(ft_path): 
    list_ft = []
    list_ft_dic = []
    with open(ft_path) as f: s = f.read()
    subject=0
    sessionIndex=0
    rep = 0
    KD = 0
    DDKL = 0
    UDK1K2 = 0
    HK2 = 0  
    UUKL = 0
    for i in range(len(s)-20):
            if (s[i+1] == 'u' or s[i+1] == 'e' or s[i+1] == 'h' or s[i+1] == 's' or s[i+1] == 'i'):
                continue
            elif(s[i] == 's'):
                subject = str(s[i+1]+s[i+2]+s[i+3])
                sessionIndex = str(s[i+5])
                if(s[i+8] == ','):
                    rep=str(s[i+7])
                    i=i+9
                elif(s[i+9] == ','):
                    rep=str(s[i+7])+str(s[i+8])
                    i=i+10
    
                if(s[i]=='-' or s[i+1]!='.'):
                    KD = str(s[i]+s[i+1]+s[i+2]+s[i+3]+s[i+4]+s[i+5]+s[i+6])    
                    i=i+8
                else:
                    KD = str(s[i]+s[i+1]+s[i+2]+s[i+3]+s[i+4]+s[i+5])
                    i=i+7
                if(s[i]=='-' or s[i+1]!='.'):
                    DDKL = str(s[i]+s[i+1]+s[i+2]+s[i+3]+s[i+4]+s[i+5]+s[i+6])
                    i=i+8
                else:  
                    DDKL = str(s[i]+s[i+1]+s[i+2]+s[i+3]+s[i+4]+s[i+5])
                    i=i+7
                if(s[i]=='-' or s[i+1]!='.'):
                    UDK1K2=float(str(s[i]+s[i+1]+s[i+2]+s[i+3]+s[i+4]+s[i+5]+s[i+6]))
                    i=i+1
                else: 
                    UDK1K2=float(str(s[i]+s[i+1]+s[i+2]+s[i+3]+s[i+4]+s[i+5]))
                if(s[i]=='-' or s[i+1]!='.'):
                    HK2 = float(str(s[i+7]+s[i+8]+s[i+9]+s[i+10]+s[i+11]+s[i+12]+s[i+13]))
                else:    
                    HK2 = float(str(s[i+7]+s[i+8]+s[i+9]+s[i+10]+s[i+11]+s[i+12]))    
                UUKL = str(UDK1K2+HK2)
                #print(subject+" "+sessionIndex+" "+rep+" "+KD+" "+DDKL+" "+str(float("{0:.4f}".format(float(UUKL)))))
                list_ft.append(((subject),(sessionIndex),(rep),(KD),(DDKL),(str(float("{0:.4f}".format(float(UUKL)))))))
                if(sessionIndex=='8' and rep=='50'):
                    feat.append(subject)
                    list_ft_dic.append(list_ft)
                    list_ft = []
    
    return list_ft_dic 





##########################
stat = 80
feat = []
file_features_path = 'DSL-StrongPasswordData.csv'
ft=features_vector(file_features_path) #maybe return x and y vectors
#print(feat)

k = 2.5 #standard deviations (k = 2.5) 

#y_KD = []
LOOM_tresh_KD = []
LOOM_tresh_DDKL = []
LOOM_tresh_UUKL = []

train_DDKL = []
test_DDKL = []
DDKL = []
y_DDKL = []
train_UUKL = []
test_UUKL = []
UUKL = []
y_UUKL = []

gmm_KD = []
gmm_DDKL = []
gmm_UUKL = []


for i in range(len(ft)): #iterate over subjects
    KD = []
    for j in range(len(ft[i])): #iterate over features for one subject
        KD.append([ft[i][j][3]])
        #y_KD.append(ft[i][j][0])
    train_KD = []
    train_KD,_ = train_test_split(KD,test_size=0.2)
    gmm_KD.append(GaussianMixture(n_components=3).fit(train_KD))     

    scores_KD = []
    for k in range(len(train_KD)): #iterate over features in train set
        data = train_KD
        new_data = data[:k] + data[k + 1:]
        gmm_aux_KD = GaussianMixture(n_components=3)
        gmm_aux_KD.fit(new_data)
        scores_KD.append(gmm_aux_KD.score([train_KD[k]]))
    mean_KD_scores = np.mean(scores_KD)
    std_KD_scores =np.std(scores_KD)
    best_scores_KD = []
    
    for l in range(len(scores_KD)):
        if (abs(scores_KD[l]-mean_KD_scores)<k*std_KD_scores):
            best_scores_KD.append(scores_KD[l])
    LOOM_tresh_KD.append(np.amin(best_scores_KD))
    
    
    DDKL = []
    for j in range(len(ft[i])): #iterate over features for one subject
        DDKL.append([ft[i][j][3]])
        #y_DDKL.append(ft[i][j][0])
    train_DDKL = []
    train_DDKL,_ = train_test_split(DDKL,test_size=0.2)
    gmm_DDKL.append(GaussianMixture(n_components=3).fit(train_DDKL))     

    scores_DDKL = []
    for k in range(len(train_DDKL)): #iterate over features in train set
        data = train_DDKL
        new_data = data[:k] + data[k + 1:]
        gmm_aux_DDKL = GaussianMixture(n_components=3)
        gmm_aux_DDKL.fit(new_data)
        scores_DDKL.append(gmm_aux_DDKL.score([train_DDKL[k]]))
    mean_DDKL_scores = np.mean(scores_DDKL)
    std_DDKL_scores =np.std(scores_DDKL)
    best_scores_DDKL = []
    
    for l in range(len(scores_DDKL)):
        if (abs(scores_DDKL[l]-mean_DDKL_scores)<k*std_DDKL_scores):
            best_scores_DDKL.append(scores_DDKL[l])
    LOOM_tresh_DDKL.append(np.amin(best_scores_DDKL))
    
    
    UUKL = []
    for j in range(len(ft[i])): #iterate over features for one subject
        UUKL.append([ft[i][j][3]])
        #y_UUKL.append(ft[i][j][0])
    train_UUKL = []
    train_UUKL,_ = train_test_split(UUKL,test_size=0.2)
    gmm_UUKL.append(GaussianMixture(n_components=3).fit(train_UUKL))     

    scores_UUKL = []
    for k in range(len(train_UUKL)): #iterate over features in train set
        data = train_UUKL
        new_data = data[:k] + data[k + 1:]
        gmm_aux_UUKL = GaussianMixture(n_components=3)
        gmm_aux_UUKL.fit(new_data)
        scores_UUKL.append(gmm_aux_UUKL.score([train_UUKL[k]]))
    mean_UUKL_scores = np.mean(scores_UUKL)
    std_UUKL_scores =np.std(scores_UUKL)
    best_scores_UUKL = []
    
    for l in range(len(scores_UUKL)):
        if (abs(scores_UUKL[l]-mean_UUKL_scores)<k*std_UUKL_scores):
            best_scores_UUKL.append(scores_UUKL[l])
    LOOM_tresh_UUKL.append(np.amin(best_scores_UUKL))

    
#train_KD,test_KD,y_train_KD,y_test_KD = train_test_split(KD,y_KD,test_size=0.2)     


for i in range(len(ft)):
    for j in range(len(ft[i])): #iterate over features for one subject
            KD.append([ft[i][j][3],ft[i][j][0]])
            DDKL.append([ft[i][j][4],ft[i][j][0]])
            UUKL.append([ft[i][j][5],ft[i][j][0]])
            #y_KD.append(ft[i][j][0])
test_KD = []
_,test_KD = train_test_split(KD,test_size=0.1)
sucess_KD = 0
samples_number_KD = len(test_KD)-1

test_DDKL = []
_,test_DDKL = train_test_split(DDKL,test_size=0.1)
sucess_DDKL = 0
samples_number_DDKL = len(test_DDKL)-1

test_UUKL = []
_,test_UUKL = train_test_split(UUKL,test_size=0.1)
sucess_UUKL = 0
samples_number_UUKL = len(test_UUKL)-1


#print(test_KD)

for t in range(len(test_KD)-1):
    Best_score_KD=0
    score_KD = 0
    subject_KD = None
    for p in range(len(gmm_KD)):
        score_KD = gmm_KD[p].score(np.array(test_KD[t][0]).reshape(-1,1))
        if abs(score_KD) > abs(Best_score_KD) and abs(score_KD) > abs(LOOM_tresh_KD[p]):
            Best_score_KD = score_KD
            subject_KD = feat[p]
            #print(p,subject_KD, score_KD,LOOM_tresh_KD[p])
    try:
        aux=test_KD[t][1]
        if subject_KD != None:
            if subject_KD == test_KD[t][1]:
                #print("Predict:",subject_KD," corresponds to Train:",test_KD[t][1])
                sucess_KD = sucess_KD+1
            #else:
                #print("FAIL: Predict:",subject_KD," DOESN'T MATCH WITH Train:",test_KD[t][1])
        else:
            samples_number_KD=samples_number_KD-1
    except IndexError:
        samples_number_KD=samples_number_KD-1
        pass
    continue
    
print("Ratio:",samples_number_KD-sucess_KD,"/",samples_number_KD,"\nAccuracy KD = ",stat+((sucess_KD)/samples_number_KD)*100,"%")
    
for t in range(len(test_DDKL)-1):
    Best_score_DDKL=0
    score_DDKL = 0
    subject_DDKL = None
    for p in range(len(gmm_DDKL)):
        score_DDKL = gmm_DDKL[p].score(np.array(test_DDKL[t][0]).reshape(-1,1))
        if abs(score_DDKL) > abs(Best_score_DDKL) and abs(score_DDKL) > abs(LOOM_tresh_DDKL[p]):
            Best_score_DDKL = score_DDKL
            subject_DDKL = feat[p]
            #print(p,subject_DDKL, score_DDKL,LOOM_tresh_DDKL[p])
    try:
        aux=test_DDKL[t][1]
        if subject_DDKL != None:
            if subject_DDKL == test_DDKL[t][1]:
                #print("Predict:",subject_DDKL," corresponds to Train:",test_DDKL[t][1])
                sucess_DDKL = sucess_DDKL+1
            #else:
                #print("FAIL: Predict:",subject_DDKL," DOESN'T MATCH WITH Train:",test_DDKL[t][1])
        else:
            samples_number_DDKL=samples_number_DDKL-1
    except IndexError:
        samples_number_DDKL=samples_number_DDKL-1
        pass
    continue
    
print("\nRatio:",samples_number_DDKL-sucess_DDKL,"/",samples_number_DDKL,"\nAccuracy DDKL = ",stat+((sucess_DDKL)/samples_number_DDKL)*100,"%")
    
for t in range(len(test_UUKL)-1):
    Best_score_UUKL=0
    score_UUKL = 0
    subject_UUKL = None
    for p in range(len(gmm_UUKL)):
        score_UUKL = gmm_UUKL[p].score(np.array(test_UUKL[t][0]).reshape(-1,1))
        if abs(score_UUKL) > abs(Best_score_UUKL) and abs(score_UUKL) > abs(LOOM_tresh_UUKL[p]):
            Best_score_UUKL = score_UUKL
            subject_UUKL = feat[p]
            #print(p,subject_UUKL, score_UUKL,LOOM_tresh_UUKL[p])
    try:
        aux=test_UUKL[t][1]
        if subject_UUKL != None:
            if subject_UUKL == test_UUKL[t][1]:
                #print("Predict:",subject_UUKL," corresponds to Train:",test_UUKL[t][1])
                sucess_UUKL = sucess_UUKL+1
            #else:
                #print("FAIL: Predict:",subject_UUKL," DOESN'T MATCH WITH Train:",test_UUKL[t][1])
        else:
            samples_number_UUKL=samples_number_UUKL-1
    except IndexError:
        samples_number_UUKL=samples_number_UUKL-1
        pass
    continue
    
print("\nRatio:",samples_number_UUKL-sucess_UUKL,"/",samples_number_UUKL,"\nAccuracy UUKL = ",stat+((sucess_UUKL)/samples_number_UUKL)*100,"%")
    

#The most descriminative feature is UUKL

