
# coding: utf-8

# In[712]:

'''
Created on December 22 2017
@auther: Yi Cao
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import gc
import copy

'''pre_clean is designed to do pre-cleaning for any structured dataset. The output dataset
    is ready to be fit in a quick and dirty model

For numerical variables, it will:
    1.remove variables when its variance is below threshold value
    2.remove highly correlated variables and keep only 1 of them
    3.impute missing
    
For categorical variables, depending on unique levels, it will:
    1. Create dummy variables for categorical variables where the count of unique level 
    is smaller than threshold
    2. Convert categorical variable to numerical if count of unique level is larger than 
    threshold. Target mean from cross validation is used for numerical conversion

It can also keep the pre-cleaning process and apply to a new dataset.
'''

class pre_clean:
    
    '''Input parameters
       data: data frame containing all predictors + id variable + target
       target: target variable name, string
       id_var: id variable name, string
       impute_miss: if impute_miss=0, it will assume there is no missing in data
                    if impute_miss=1, it will impute missing categorical with 'mis' 
                    and will use median to impute numerical variable 
                    values other than 0/1 are not accepted
                    if total missing rate for numerical is >20%, it will create another dummy indicator 
                    where 1 = missing
    '''
    
    def __init__(self,data,target,id_var,impute_miss=0):
        
        '''all predictors inlcuding id variable'''
        self.all_predictor = [x for x in data.columns if x not in target]
        '''0 assume there is no missing in the dataset, 1 will impute missing'''
        self.impute_miss = impute_miss
        self.data = copy.deepcopy(data)
        self.id_var = id_var
        self.target = target
        self.cat_list = list()
        self.num_list = list()
              
        if impute_miss==0:

            self.cat_list = [x for x in self.data.columns if x not in [target,id_var] and 
                            self.data.dtypes[x]=='O']
            self.num_list = [x for x in self.data.columns if x not in [target,id_var] and 
                            self.data.dtypes[x]!='O']       
        elif impute_miss==1:        
            self.mis_dummy = list()
            
            for col in self.data.columns:
                if self.data.dtypes[col] == 'O' and col not in [target,id_var]:
                    self.cat_list.append(col)
                    self.data[col].fillna('mis',inplace=True)
                elif self.data.dtypes[col] != 'O' and col not in [target,id_var]:
                    self.num_list.append(col)
                    if self.data[col].isnull().sum()>np.float(self.data.shape[0]/5):
                        mis_ind = str(col)+'_mis_ind'
                        self.mis_dummy.append(col)
                        self.data[mis_ind] = self.data[col].map(lambda x: 1 if np.isnan(x)==True else 0)
                        self.data[col].fillna(self.data[col].median(),inplace=True)  
                    else:
                        self.data[col].fillna(self.data[col].median(),inplace=True)  
        else:
            print('impute_miss can only be 1 or 0')
            
    '''do numerical variable cleaning'''
    '''    
    Input paramter:
    corr_th: float, pearson correlation coeffecient threshold, any pair of num-variables with greater than
             threshold coefficient, only one variable will be kept
    var_th : float, any standard variable with variance lower than threshold will be removed, variables are
            rescled before comparing its variance to threshold.
    '''
    def num_clean(self,corr_th,var_th):
        
        '''remove_num_var is the list containing variable removed due to low variance'''
        self.remove_num_var = list()
        scaler = MinMaxScaler((-1,1))
        for col in self.num_list:
            if np.var(scaler.fit_transform(self.data[col].reshape(-1,1)))<var_th:
                self.remove_num_var.append(col)

        '''remove_corr is the list containing variables removed due to high-correlated with other
        variables'''
        tmp = self.data[[x for x in self.num_list if x not in self.remove_num_var]]       
        corr = np.corrcoef(tmp,rowvar=0)     
        self.remove_corr = list()
        ncol = corr.shape[1]     
        for i in range(ncol):
            if i in self.remove_corr:
                continue
            else:
                self.remove_corr.extend(tmp.columns[[x for x in range(i+1,ncol) if 
                                        ((corr[i,x]>corr_th)|(corr[i,x]<corr_th*(-1)))]])
        self.remove_corr = list(set(self.remove_corr))
        
        '''clean up temporary files'''
        del corr,ncol,tmp
        gc.collect()
        
        self.data.drop(list(set(self.remove_num_var + self.remove_corr)),axis=1,inplace=True)
     
    '''do categorical variable cleaning'''
    '''    
    Input parameter:
    cat_num_level_th: for one categorical variable, if its unique level count is greater than cat_num_level_th,
        it will be converted to numerical. Otherwise, it will be dummied. After cleanning, original variable will
        be dropped and converted/dummied varaibles will be added
    '''
    def cat_clean(self,cat_num_level_th):
        
        self.cat_onehot = [x for x in self.cat_list if len(self.data[x].unique())< cat_num_level_th]
        self.cat_num = [x for x in self.cat_list if x not in self.cat_onehot]
        self.cat_num_lkup = pd.DataFrame(columns=['var','level','target'])
        self.onehot_lkup = pd.DataFrame(columns=['var','level'])
        
        for col in self.cat_num:
            cat_out = str(col)+str('_num')
            cat_num_tmp = cvt_cat_num(self.data,col,cat_out,self.id_var,self.target
                                      ,nfolds=5,r1=0.4,r2=0.6)
            self.data = pd.merge(self.data,cat_num_tmp,on=self.id_var,how='inner')
            unique_level = len(data[col].unique())
            lkup = self.data.groupby(col)[self.target].mean()
            self.cat_num_lkup = pd.concat([self.cat_num_lkup,pd.DataFrame({'var':col,'level':lkup.index,'target':lkup.values})],axis=0)          
            self.data.drop(col,axis=1,inplace=True) 
        
        if len(self.cat_onehot)>0:  
            dummy = pd.get_dummies(self.data[[self.id_var]+self.cat_onehot])
            for col in self.cat_onehot:
                self.onehot_lkup = pd.concat([self.onehot_lkup,pd.DataFrame({'var':col,'level':self.data[col].unique()})],axis=0)
            self.data.drop(self.cat_onehot,axis=1,inplace=True)
            self.data = pd.merge(self.data,dummy,on=self.id_var,how='inner')
            
            del dummy
            gc.collect()
    
    '''
    fit_new_data function will apply the saved previous fit data preprocessing to new_data directly. 
    Demon code can be found below
    '''
    '''
    Input parametre:
    new_data: a new dataset to be fit using exsiting pre-clean results
    '''
    def fit_new_data(self,new_data):
        
        '''
        check if new data contain all necessary predictors and if categorical variables has new levels
        '''       
        assert any(len(x)>0 for x in [self.remove_num_var, 
                                     self.remove_corr,
                                     self.cat_onehot,
                                     self.cat_num]), "There is no cleaning"
        assert (set(new_data.columns) >= set(self.all_predictor)), "New data does not contain all predictors"
        
        for col in self.cat_onehot:
            assert set(list(self.onehot_lkup[self.onehot_lkup['var']==col]['level'].values))==set(list(new_data[col].unique())),"New data has different level for " +(col)
                
        self.new_data = copy.deepcopy(new_data) 
        
        if self.impute_miss ==1:
            for col in self.num_list:
                if col in self.mis_dummy:
                    self.new_data[col+'_mis_ind'] = self.new_data[col].map(lambda x: 1 if np.isnan(x)==True else 0)
                self.new_data[col].fillna(self.new_data[col].median(),inplace=True)
        
        if len(self.remove_num_var)>0:
            self.new_data.drop(self.remove_num_var,axis=1,inplace=True)
        if len(self.remove_corr)>0:
            self.new_data.drop(self.remove_corr,axis=1,inplace=True)
        if len(self.cat_num)>0:
            for col in self.cat_num:
                var_out = str(col)+'_num'
                lkup = self.cat_num_lkup[self.cat_num_lkup['var']==col].drop('var',axis=1)
                lkup=lkup.rename(columns = {'level':col,'target':var_out})
                self.new_data = pd.merge(self.new_data,lkup,on=col,how='left')
                self.new_data.drop(col,axis=1,inplace=True)
                self.new_data[var_out].fillna(self.new_data[var_out].median(),inplace=True)
                
        if len(self.cat_onehot)>0:
            dummy = pd.get_dummies(self.new_data[self.cat_onehot+[self.id_var]])
            self.new_data.drop(self.cat_onehot,axis=1,inplace=True)
            self.new_data = pd.merge(self.new_data,dummy,on=self.id_var,how='inner')
    
'''Create fold indexes for a given data frame'''
'''Input parameter:
    data: data frame to be cut into folds
    nfolds: int, number of folds
'''
'''Output: pandas series with folds indexes'''
    
def cre_folds(data,nfolds):
    nrow = data.shape[0]
    folds = list(range(nfolds))*int(np.ceil(float(nrow)/nfolds))
    folds = folds[:nrow]
    np.random.shuffle(folds)
    folds = pd.Series(folds)
    return folds

'''if a categorical will be converted to numerical, I will use cross-validation
target mean to do imputation, cvt_cat_num is the function to calculate target mean
given cross-validation results'''
'''
Input parameter:
data: data frame containing id,target and the categorical variable to be converted
cat: string, categorical variable name
cat_out: string, variable name for converted categorical variable
id_var: string, id variable name
target: string, target variable name
nfolds: number of folds the target mean is coming from
r1: coefficient for validation overall fold mean
r2: coefficient for validation specific level mean
for a 5 fold cut, the values for categorical variable in the first fold is
r1*(other 4 fold target mean) +r2*(other 4 fold target mean at specific level).
'''
'''Output: dataframe with id and cat_out'''
    
def cvt_cat_num(data,cat,cat_out,id_var,target,nfolds=5,r1=0.4,r2=0.6):

    tmp = data[[id_var,target,cat]]
    df_out = pd.DataFrame({id_var:[],cat_out:[]})
    folds = cre_folds(tmp,nfolds=nfolds)

    for i in range(nfolds):
        tr = tmp.ix[folds.values == i,[id_var,cat]]
        va = tmp.ix[folds.values != i,[cat,target]] 
        fold_mean = va[target].mean()
        va = va.groupby(cat)[target].mean()
        #print(va)
        va_lkup = pd.DataFrame({cat:va.index, cat_out:va.values})
        va_lkup[cat_out] = r1*va_lkup[cat_out] + r2*fold_mean

        tr = pd.merge(tr,va_lkup,on=cat,how='left')
        tr = tr[[id_var,cat_out]]
        df_out = df_out.append(tr)

    df_out = df_out.fillna(data[target].mean())
    return df_out



    


# In[713]:

if __name__=="__main__":
    '''
    Use Boston dataset to test the tool
    '''
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    Boston = datasets.load_boston()
    data = pd.DataFrame(Boston.data,columns=Boston.feature_names)
    data['target'] = Boston.target

    '''Add categorical variables in order to test tool'''
    data['CAT_CHAS'] = data['CHAS'].map(lambda x: 'Yes' if x==1 else 'No' )
    data['CAT_CHAS1'] = data['CHAS'].map(lambda x: 'Yes' if x==1 else 'No' )
    data['CAT_RAD'] = data['RAD'].map(lambda x: str(x))
    data['CAT_RAD1'] = data['RAD'].map(lambda x: str(x))

    '''Add ID cariable'''
    data['ID'] = np.arange(data.shape[0])
    train,test = train_test_split(data,test_size=0.3,random_state=777)
    
    '''numerical cleaning'''
    test_tool = pre_clean(train,'target','ID',impute_miss=0)
    
    '''remove variables with <0.05 variance or >0.9 correlation with other predicotrs'''
    test_tool.num_clean(0.9,0.05)
    
    ''' categorical cleaning, for unique level<5 cat var, do onehot encoding. for >5 unique level
    cat variable, convert them to numerical variables, new variable will add '_num' to diffrientiate'''
    test_tool.cat_clean(5)
    
    '''test new dataset'''
    test_tool.fit_new_data(test)
    
    '''check output
    removed variable list due to low variance'''
    print(test_tool.remove_num_var)
    
    '''removed variable list due to high correlation'''
    print(test_tool.remove_corr)
    
    '''dummied categorical variable list'''
    print(test_tool.cat_onehot)
    
    '''converted to numerical variables'''
    print(test_tool.cat_num)
    
    '''compare train and test output'''
    print(test_tool.data.head(5))
    print(test_tool.new_data.head(5))
    
    
    


# In[ ]:



