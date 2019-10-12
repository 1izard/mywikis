# %%
import numpy as np 
import numpy.random as random
import scipy as sp 
import pandas as pd 
from pandas import Series, DataFrame

import matplotlib.pyplot as plt 
import matplotlib as mpt 
import seaborn as sns 
%matplotlib inline

%precision 3
# %%
hier_df = DataFrame(
    np.arange(9).reshape(3, 3),
    index = [
        ['a', 'a', 'b'],
        [1, 2, 2]
    ],
    columns = [
        ['Osaka', 'Tokyo', 'Osaka'],
        ['Blue', 'Red', 'Red']
    ]
)
hier_df
#%%
hier_df.index.names = ['key1', 'key2']
hier_df.columns.names = ['city', 'color']
hier_df
#%%
hier_df['Osaka']
#%%
hier_df.sum(level='key2', axis=0)
#%%
hier_df.sum(level='color', axis=1)
#%%
hier_df.drop(['b'])
#%%
hier_df
#%%
hier_df1 = DataFrame(
    np.arange(12).reshape((3, 4)),
    index = [['c', 'd', 'd'], [1, 2, 1]],
    columns = [
        ['Kyoto', 'Nagoya', 'Hokkaido', 'Kyoto'],
        ['Yellow', 'Yellow', 'Red', 'Blue']
    ]
)

hier_df1.index.names = ['key1', 'key2']
hier_df1.columns.names = ['city', 'color']
hier_df1
#%%
hier_df1.mean(level='city', axis=1)
#%%

#%%
hier_df1.sum(level='key2', axis=0)
#%%
data1 = {
    'id' : ['100', '101', '102', '103', '104', '106', '108', '110', '111', '113'],
    'city' : ['Tokyo', 'Osaka', 'Kyoto', 'Hokkaido', 'Tokyo', 'Tokyo', 'Osaka', 'Kyoto', 'Hokkaido', 'Tokyo'],
    'birth_year' : [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
    'name' : ['Hiroshi', 'Akiko', 'Yuki', 'Satoru', 'Steeve', 'Mituru', 'Aoi', 'Tarou', 'Suguru', 'Mitsuo']
}
df1 = DataFrame(data1)
df1
#%%
data2 = {
    'id' : ['100', '101', '102', '105', '107'],
    'math' : [50, 43, 33, 76, 98],
    'english' : [90, 30, 20, 50, 30],
    'sex' : ['M', 'F', 'F', 'M', 'M'],
    'index_num' : [0, 1, 2, 3, 4]
}
df2 = DataFrame(data2)
df2
#%%
print('Joint Table')
pd.merge(df1, df2, on='id')
#%%
pd.merge(df1, df2, how='outer')
#%%
pd.merge(df1, df2, left_index=True, right_on='index_num')
#%%
pd.merge(df1, df2, how='left')
#%%
data3 = {
    'id' : ['117', '118', '119', '120', '125'],
    'city' : ['Chiba', 'Kanagawa', 'Tokyo', 'Fukuoka', 'Okinawa'],
    'birth_year' : [1990, 1989, 1992, 1997, 1982],
    'name' : ['Suguru', 'Kouchi', 'Satochi', 'Yukie', 'Akari']
}
df3 = DataFrame(data3)
df3
#%%
concat_data = pd.concat([df1, df3])
concat_data
#%%
data4 = {
    'id' : ['0', '1', '2', '3', '4', '6', '8', '11', '12', '13'],
    'city' : ['Tokyo', 'Osaka', 'Kyoto', 'Hokkaido', 'Tokyo', 'Tokyo', 'Osaka', 'Kyoto', 'Hokkaido', 'Tokyo'],
    'birth_year' : [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
    'name' : ['Hiroshi', 'Akiko', 'Yuki', 'Satoru', 'Steeve', 'Mituru', 'Aoi', 'Tarou', 'Suguru', 'Mitsuo']
}
df4 = DataFrame(data4)
df4
#%%
data5 = {
    'id' : ['0', '1', '3', '6', '8'],
    'math' : [20, 30, 50, 70, 90],
    'english' : [30, 50, 50, 70, 20],
    'sex' : ['M', 'F', 'F', 'M', 'M'],
    'index_num' : [0, 1, 2, 3, 4]
}
df5 = DataFrame(data5)
df5
#%%
inner_joint = pd.merge(df4, df5)
inner_joint
#%%
all_joint = pd.merge(df4, df5, how='outer')
all_joint
#%%
data6 = {
    'id' : ['70', '80', '90', '120', '150'],
    'city' : ['Chiba', 'Kanagawa', 'Tokyo', 'Fukuoka', 'Okinawa'],
    'birth_year' : [1980, 1999, 1995, 1994, 1994],
    'name' : ['Suguru', 'Kouichi', 'Satochi', 'Yukie', 'Akari']
}
df6 = DataFrame(data6)
df6
#%%
vertical_joint = pd.concat([df4, df6])
vertical_joint
#%%
hier_df = DataFrame(
    np.arange(9).reshape((3, 3)),
    index=[
        ['a', 'a', 'b'],
        [1, 2, 2]
    ],
    columns=[
        ['Osaka', 'Tokyo', 'Osaka'],
        ['Blue', 'Red', 'Red']
    ]
)
hier_df
#%%
hier_df.stack()
#%%
hier_df.stack().unstack()
#%%
dupli_data = DataFrame({
    'col1' : [1, 1, 2, 3, 4, 4, 6, 6],
    'col2' : ['a', 'b', 'b', 'b', 'c', 'c', 'b', 'b']
})
print('Original data')
dupli_data
#%%
dupli_data.duplicated()
#%%
dupli_data.drop_duplicates()
#%%
city_map = {
    'Tokyo' : 'Kanto',
    'Hokkaido' : 'Hokkaido',
    'Osaka' : 'Kansai',
    'Kyoto' : 'Kansai'
}
city_map
#%%
df1['region'] = df1['city'].map(city_map)
df1
#%%
df1['up_two_num'] = df1['birth_year'].map(lambda x: str(x)[0:3])
df1
#%%
birth_year_bins = [1980, 1985, 1990, 1995, 2000]

birth_year_cut_data = pd.cut(df1.birth_year, birth_year_bins)
birth_year_cut_data
#%%
pd.value_counts(birth_year_cut_data)
#%%
type(birth_year_cut_data)
#%%
group_names = ['early1980s', 'late1980s', 'early1990s', 'late1990s']
birth_year_cut_data = pd.cut(df1.birth_year, birth_year_bins, labels=group_names)
pd.value_counts(birth_year_cut_data)

#%%
pd.cut(df1.birth_year, 2)
#%%
pd.value_counts(pd.qcut(df1.birth_year, 2))
#%%
student_data_math = pd.read_csv('/Users/tsubasa/Jupyter/20190916_toudai/chap3/student-mat.csv', sep=';')
student_data_math['double age'] = student_data_math['age'].map(lambda x: 2*x)
student_data_math
#%%
absences_bins = [0, 1, 5, 100]
student_data_math_cut = pd.cut(student_data_math.absences, absences_bins, right=False)
pd.value_counts(student_data_math_cut)
#%%
student_data_math_qcut = pd.qcut(student_data_math.absences, 3)
pd.value_counts(student_data_math_qcut)
#%%
df1
#%%
df1.groupby('city').size()
#%%
df1.groupby('city')['birth_year'].mean()
#%%
df1.groupby(['region', 'city'])['birth_year'].mean()
#%%
df1.groupby(['region', 'city'], as_index=False)['birth_year'].mean()
#%%
for group, subdf in df1.groupby('region'):
    print('-' * 30)
    print('Region Name:{}'.format(group))
    print(subdf)
#%%
student_data_math = pd.read_csv('/Users/tsubasa/Jupyter/20190916_toudai/chap3/student-mat.csv', sep=';')

functions = ['count', 'mean', 'max', 'min']
grouped_student_math_data1 = student_data_math.groupby(['sex', 'address'])
grouped_student_math_data1['age', 'G1'].agg(functions)
#%%
student_data_math.groupby('school').G1.mean()
#%%
student_data_math.groupby(['school', 'sex'])['G1', 'G2', 'G3'].mean()
#%%
student_data_math.groupby(['school', 'sex'])['G1', 'G2', 'G3'].agg(('max', 'min'))
#%%
import numpy as np
from numpy import nan as NA 
import pandas as pd 

df = pd.DataFrame(np.random.rand(10, 4))

df.iloc[1, 0] = NA
df.iloc[2:3, 2] = NA
df.iloc[5:, 3] = NA
#%%
df
#%%
df.dropna()
#%%
df[[0, 1]].dropna()
#%%
df.fillna(0)
#%%

#%%
