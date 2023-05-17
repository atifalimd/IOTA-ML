import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
from application_log import logger

class DataCollection:
    """
    This class shall be used for reading dataset.

    Written By : Atif Ali Mohammed
    """

    def __init__(self,path):
        self.path = path
        self.log_writer = logger.AppLogger()


    def reading_dataframe(self):
        """
        Method Name : reading_dataframe
        Description : This method read and return the dataset from the given path.
        On Failure  : Raise OSError, exception

        Written By  : Atif Ali Mohammed
        """
        try:
            df = pd.read_csv(self.path)
            return df
        
        except OSError:
            with open("log/reading_dataframe_log.txt",'a+') as file:
                self.log_writer.log(file,'Error occurred while reading the dataset :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open("log/reading_dataframe_log.txt",'a+') as f:
                self.log_writer.log(f,"Error Occured :: {}".format(e))
            raise e



df = DataCollection(path='csv_files/postcodes.csv')
postcode = df.reading_dataframe()


class SelectColumns:
    """
    This class shall be used to select the columns based on object type.
    Written By: Atif Ali Mohammed
    """
    def __init__(self,df):
        self.df=df
        self.log_writer=logger.AppLogger()
    
    def select_numerical_cols(self):
        """
        Method Name : select_numerical_cols
        Description : This method return numerical columns only
        On Failure  : Raise OSError, exception
        Written By  : Atif Ali Mohammed
        """
        try:
            df_num= self.df.select_dtypes(exclude='object')
            return df_num
        except OSError:
            with open('log/select_numerical_cols.txt', 'a+') as file:
                self.log_writer.log(file,'Error Occurred while selecting numerical columns :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open('log/select_numerical_cols.txt', 'a+') as f:
                self.log_writer.log(f,'Error Occurred while selecting numerical columns :: {}'.format(e))
            raise e
             
df_num=SelectColumns(df=postcode)
postcode_num = df_num.select_numerical_cols()

cat_df = DataCollection(path='csv_files/cat-population-per-postcode-district-upper-95th-percentile-1.csv')
cat_df = cat_df.reading_dataframe()
dog_df = DataCollection(path='csv_files/dogs-per-household-per-postcode-district-lower-95th-percentile-1.csv')
dog_df = dog_df.reading_dataframe()
ons_df = DataCollection(path='csv_files/ONS_Age_House_Data.csv')
ons_df = ons_df.reading_dataframe()


class AddingColumns:
    """
    This class shall be used to add required non numerical columns to the dataset
    Written By : Atif Ali Mohammed
    """
    def __init__(self,df):
        self.log_writer = logger.AppLogger()
        self.df = df
    def adding_part_postcode(self,columns):
        """
        Method Name : adding_part_postcode
        Description : Adding 'Postcode area','Postcode district'to the postcode_num
        Written BY  : Atif Ali Mohammed
        """
        try:
            part_df = self.df[columns]
            return part_df
        except OSError as oe:
            with open('log/adding_part_postcode.txt','a+') as file:
                self.log_writer.log(file,'Error occurred while adding columns to the dataset :: {}'.format(oe))
        except Exception as e:
            with open('log/adding_part_postcode.txt','a+') as f:
                self.log_writer.log(f,'Error occurred while adding columns to the dataset :: {}'.format(e))
add_col = AddingColumns(df = postcode)
part_postcode=add_col.adding_part_postcode(columns=['Postcode area','Postcode district'])


class CombiningDataFrames:
    """
    This class can be used to merge dataframes.

    Written By : Atif Ali Mohammed
    """
    def __init__(self):
        self.log_writer = logger.AppLogger()
    def merge_dataframes(self,df1,df2,left_index,right_index,left_on,right_on,on):
        """
        Method Name : merge_dataframes
        Description : Merging two datasets on partial postcode level
        Written BY  : Atif Ali Mohammed
        """
        try:
            df=pd.merge(df1,df2,left_on=left_on,right_on=right_on,left_index=left_index, right_index=right_index,on=on)
            return df 
        except OSError:
            with open('log/merge_dataframes.txt', 'a+') as file:
                self.log_writer.log(file,'Error Occurred while combining datasets :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open('log/merge_dataframes.txt', 'a+') as f:
                self.log_writer.log(f,'Error Occurred while combining datasets :: {}'.format(e))
            raise e
    

    def concatinate_dataframes(self,df1,df2):
        """
        Method Name : concatinate_dataframes
        Description : Concatinating two datasets
        Written BY  : Atif Ali Mohammed
        """
        try:
            df = pd.concat([df1,df2],axis=1)
            return df
        except OSError:
            with open('log/concatinate_dataframes.txt', 'a+') as file:
                self.log_writer.log(file,'Error Occurred while combining datasets :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open('log/concatinate_dataframes.txt', 'a+') as f:
                self.log_writer.log(f,'Error Occurred while combining datasets :: {}'.format(e))
            raise e


gathering_dataset = CombiningDataFrames()
part_postcode_num = gathering_dataset.merge_dataframes(df1=part_postcode,df2=postcode_num,left_index=True,right_index=True,left_on=None,right_on=None,on=None)
part_postcode_num_dog = gathering_dataset.merge_dataframes(df1=part_postcode_num,df2=dog_df,left_on='Postcode district',right_on='PostcodeDistrict',on=None,left_index=False,right_index=False)
part_postcode_num_dog_cat = gathering_dataset.merge_dataframes(df1=part_postcode_num_dog,df2=cat_df,left_index=False, right_index = False, left_on = None, right_on = None, on='PostcodeDistrict')

# merge_part_postcode_num_dog = CombiningDataFrames()
# part_postcode_num_dog = merge_part_postcode_num_dog.merge_dataframes(df1=part_postcode_num,df2=dog_df,left_on='Postcode district',right_on='PostcodeDistrict',on=None,left_index=False,right_index=False)

# merge_part_postcode_num_dog_cat = CombiningDataFrames()
# part_postcode_num_dog_cat = merge_part_postcode_num_dog_cat.merge_dataframes(df1=part_postcode_num_dog,df2=cat_df,left_index=False, right_index = False, left_on = None, right_on = None, on='PostcodeDistrict')

class UsefulColumns:
    """
    This class shall be used to keep only useful columns of the dataset
    """
    def __init__(self):
        self.log_writer = logger.AppLogger()

    def droping_unuseful_columns(self,df):
        """
        Method Name : droping_unuseful_columns
        Description : Dropping not important columns of the dataset 
        Written By  : Atif Ali Mohammed  
        """

        try:
            df = df.drop(columns = ['London zone','Latitude','Longitude','Easting','Northing','Postcode district','Quality','User Type'])
            return df
        
        except OSError:
            with open('log/droping_unuseful_columns.txt', 'a+') as file:
                self.log_writer.log(file,'Error Occurred while updating useful columns of datasets :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open('log/useful_columns.txt', 'a+') as f:
                self.log_writer.log(f,'Error Occurred while updating useful columns of datasets :: {}'.format(e))
            raise e
    
use_col = UsefulColumns()
part_postcode_num_dog_cat = use_col.droping_unuseful_columns(df=part_postcode_num_dog_cat)

class RenamingColumns:
    """
    This class shall be used to renaming columns.
    """
    dic={'EstimatedCatPopulation_Upper95':'cat','DogsPerHousehold_lower95':'dog'}

    columns= ['communal_est','per_heactre','0_4yrs','5_14yrs','25_44yrs','45_64yrs','65_89yrs','90yrs_over',
'16yrs_single','16yrs_married','16yrs_divorced','white','mixed','Indian',
'Pakistani','Bangladeshi','Other_asians','Black','Arab','uk_born','pre_2004_born','post_2004_born','no_spk_english','House_no_children','house_no_dependent_children',
'house_students','detached_house','semi-detached house','terrace_house','flat','own','private_rent','social_rent',
'one_room_rent','Illness_ratio','unpaid_care','edu_Level_2','edu_Level_3','edu_Level_4','students','two`-cars',
'public_transport_work','private_transport_work','walk_cycle','unemployed','part-time','full-time',
'agriculture_industry','construction_industry','storage_industry','energy_industry','retail_industry',
'transport_industry','food_industry','tech_industry','real_estate_industry','admin_industry','security_industry',
'edu_industry','social_work']
    

    def __init__(self):
            self.log_writer = logger.AppLogger()
        
    def rename_cols(self,df):
        """
        Method Name : rename_cols
        Description : Rename column of the dataset
        Written By  : Atif Ali Mohammed  
        """
        try:
            df = df.rename(columns=self.dic)
            return df
        except OSError:
            with open('log/rename_columns.txt', 'a+') as file:
                self.log_writer.log(file,'Error Occurred while renaming columns of datasets :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open('log/rename_columns.txt', 'a+') as f:
                self.log_writer.log(f,'Error Occurred while renaming columns of datasets :: {}'.format(e))
            raise e
    
    def replacing_col_names(self,df):
        """
        Method Name : replacing_col_names
        Description : Replacing column names of the dataset with the list of column names
        Written By  : Atif Ali Mohammed  
        """
        try:
            df_clean = df.iloc[:,8:-1]
            df_clean.columns=self.columns
            return df_clean
        except OSError:
            with open('log/replacing_col_names.txt', 'a+') as file:
                self.log_writer.log(file,'Error Occurred while replacing columns name of datasets :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open('log/replacing_col_names.txt', 'a+') as f:
                self.log_writer.log(f,'Error Occurred while replacing columns name of datasets :: {}'.format(e))
            raise e


rename_columns = RenamingColumns()
clean_postcode = rename_columns.rename_cols(df=part_postcode_num_dog_cat)
ons_clean = rename_columns.replacing_col_names(df=ons_df)

class CleaningData:
    """
    This class shall be used to clean data 
    """
    def __init__(self):
        self.log_writer = logger.AppLogger()
    
    def clean_column(self,df,col):
        """
        Method Name : clean_column
        Description : Make all the columns of same dtype
        Written By  : Atif Ali Mohammed
        """
        try:
            df[col] = df[col].str.replace(',','').astype(float)
            return df
        except OSError:
            with open('log/clean_column.txt', 'a+') as file:
                self.log_writer.log(file,'Error Occurred while replacing columns name of datasets :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open('log/clean_column.txt', 'a+') as f:
                self.log_writer.log(f,'Error Occurred while replacing columns name of datasets :: {}'.format(e))
            raise e
    
    def change_dtype(self,df,col):
        """
        Method Name : change_dtype
        Description : Change all the column values to float dtype
        Written By  : Atif Ali Mohammed
        """
        try:
            df[col][3:4]='42'
            df = df.astype('float16')
            return df
        except OSError:
            with open('log/change_dtype.txt', 'a+') as file:
                self.log_writer.log(file,'Error Occurred while replacing columns name of datasets :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open('log/change_dtype.txt', 'a+') as f:
                self.log_writer.log(f,'Error Occurred while replacing columns name of datasets :: {}'.format(e))
            raise e

clean_data=CleaningData()
clean_postcode = clean_data.clean_column(df=clean_postcode, col='cat')


ons_clean = clean_data.change_dtype(df = ons_clean, col='communal_est')

concat_df = CombiningDataFrames()
ons_clean = concat_df.concatinate_dataframes(df1=ons_df[['LA Code']],df2=ons_clean)
print("clean_postcode")
print(clean_postcode)

print("postcode")
print(postcode[['District Code']])

concat_df_post=CombiningDataFrames()
clean_postcode=concat_df_post.concatinate_dataframes(df1=postcode[['District Code']],df2=clean_postcode)
clean_postcode = clean_postcode[:2484173]

class RemovingOutliers:
    """
    This class shall be used to remove outliers 
    """
    def __init__(self):
        self.log_writer = logger.AppLogger()
    
    def capping_95_prct(self,df,col):
        """
        Method Name : capping_95_prct
        Description : Capping column values to 95 percentile
        Written By  : Atif Ali Mohammed
        """
        try:
            pct_95=df[col].quantile(0.95)
            df[col] = df[col].clip(None,pct_95)
            return df
        except OSError:
            with open('log/capping_95_prct.txt', 'a+') as file:
                self.log_writer.log(file,'Error Occurred while replacing columns name of datasets :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open('log/capping_95_prct.txt', 'a+') as f:
                self.log_writer.log(f,'Error Occurred while replacing columns name of datasets :: {}'.format(e))
            raise e
    

remov_out=RemovingOutliers()
clean_postcode = remov_out.capping_95_prct(df=clean_postcode,col='cat')


class FillNan():
    """
    This class shall be used to fill Nan Values.
    Written By : Atif Ali Mohammed
    """
    def __init__(self):
        self.log_writer = logger.AppLogger()
    
    def fill_null_with_mean(self,df,col,null_col):
        """
        Method Name : fill_null_with_mean
        Description : Filling null values with mean at postcode level
        Written By  : Atif Ali Mohammed
        """   
        try:
            for column in null_col:
                print(column,'fillna started')
                lst_postcode_area = list(df[col].unique())
                mean_dict = df[[col,column]].groupby(col).mean().to_dict()
                for area in lst_postcode_area:
                    idx = df[df[col]==area].index
                    df[column].iloc[idx]=mean_dict[column][area]
                    df[column] = df[column].fillna(df[column].mean())
            return df


        except OSError:
            with open('log/fill_null_with_mean', 'a+') as file:
                self.log_writer.log(file,'Error Occurred while filling nan values of the  datasets :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open('log/fill_null_with_mean', 'a+') as f:
                self.log_writer.log(f,'Error Occurred while updating filling nan values of the datasets :: {}'.format(e))
            raise e
        
fill_nan = FillNan()
postcode_popul_house_area = fill_nan.fill_null_with_mean(df=clean_postcode[['Population', 'Households','Postcode area']],
                                  col='Postcode area', null_col=['Population','Households'])

column = ['PostcodeDistrict','District Code','Altitude','Index of Multiple Deprivation','Distance to station','Average Income','Distance to sea','dog','cat']
postcode_district = clean_postcode[column]
null_col = ['Index of Multiple Deprivation','Distance to station','Average Income','Distance to sea']
fillna_postcode = fill_nan.fill_null_with_mean(df=postcode_district,null_col=null_col,col='PostcodeDistrict')


comb_clean_postcode = CombiningDataFrames()
clean_postcode = comb_clean_postcode.concatinate_dataframes(df1=postcode_popul_house_area,df2=fillna_postcode)

class GroupbyOper:
    """
    This class shall be used to perform Groupby operation.
    Written By : Atif Ali Mohammed
    """
    def __init__(self):
        self.log_writer = logger.AppLogger()
    
    def groupby_mean(self,df,col):
        """
        Method Name : groupby_mean
        Description : Grouping the dataset and caluculating mean wrt given column
        Written By  : Atif Ali Mohammed
        """
        try:
            df = df.groupby(col).mean()
            return df
        except OSError:
            with open('log/groupby_mean.txt', 'a+') as file:
                self.log_writer.log(file,'Error Occurred while replacing columns name of datasets :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open('log/groupby_mean.txt', 'a+') as f:
                self.log_writer.log(f,'Error Occurred while replacing columns name of datasets :: {}'.format(e))
            raise e

grbpy = GroupbyOper()
postcode_grbpy = grbpy.groupby_mean(df = clean_postcode,col='District Code')      
ons_grbpy = grbpy.groupby_mean(df=ons_clean,col = 'LA Code')

grpby_ons_postcode = CombiningDataFrames()
grpby_ons_postcode = grpby_ons_postcode.merge_dataframes(df1=postcode_grbpy,df2=ons_grbpy,left_index=True,right_index=True,left_on=None,right_on=None,on=None)


class StandardizeDataset:
    """
    This class shall be used to standardize the dataset
    Written By : Atif Ali Mohammed
    """
    def __init__(self):
            self.log_writer = logger.AppLogger()
        
    def scaler(self,df):
        """
        Method Name : scaler
        Description : Standardize the Dataset 
        Written By  : Atif Ali Mohammed
        """
        try:
            scale = StandardScaler()
            df = scale.fit_transform(df)
            return df

        except OSError:
            with open('log/standardize.txt', 'a+') as file:
                self.log_writer.log(file,'Error Occurred while standardizing dataset :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open('log/standardize.txt', 'a+') as f:
                self.log_writer.log(f,'Error Occurred while standardizing dataset :: {}'.format(e))
            raise e

scale = StandardizeDataset()
scale_grpby_ons_postcode = scale.scaler(df = grpby_ons_postcode)




class DimensionalityReduction:
    """
    This class shall be use to reduce the dimensions of the dataset
    Written By : Atif Ali Mohammed
    """

    def __init__(self):
            self.log_writer = logger.AppLogger()
        
    def pca(self,n_components,df, pca_cols):
        """
        Method Name : pca
        Description : It helps in selecting desired number of pca columns
        Written By  : Atif Ali Mohammed
        """
        try:
            pca=PCA(n_components=n_components)
            df_pca = pca.fit_transform(df)
            df_pca=pd.DataFrame(df_pca)
            df_pca.columns=pca_cols
            return df_pca
        except OSError:
            with open('log/dimensionality_reduction.txt', 'a+') as file:
                self.log_writer.log(file,'Error Occurred while reducing dimensions of datasets :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open('log/dimensionality_reduction.txt', 'a+') as f:
                self.log_writer.log(f,'Error Occurred while rreducing dimensions of datasets :: {}'.format(e))
            raise e

pca_transform = DimensionalityReduction()
pca_transform = pca_transform.pca(df = scale_grpby_ons_postcode,n_components=2,pca_cols=['pca_1','pca_2'])
print("pca_transform")
print(pca_transform)

class CalculateCluster:
    """
    This class shall be use to calculate appropriate number of clusters
    Written By : Atif Ali Mohammed
    """

    def __init__(self):
            self.log_writer = logger.AppLogger()

    def forming_cluster(self,df,n_clusters):
        """
        Method Name : forming_cluster
        Description : It helps in forming clusters using kmeans
        Written By  : Atif Ali Mohammed
        """
        try:
            kmeans_cluster=KMeans(n_clusters=n_clusters,random_state=42)
            df['cluster']=kmeans_cluster.fit_predict(df)
        except OSError:
            with open('log/calculate_cluster.txt', 'a+') as file:
                self.log_writer.log(file,'Error Occurred while reducing dimensions of datasets :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open('log/calculate_cluster.txt', 'a+') as f:
                self.log_writer.log(f,'Error Occurred while rreducing dimensions of datasets :: {}'.format(e))
            raise e


    def calculate_centroid(self,df,n_clusters):
        """
        Method Name : calculate_centroid
        Description : It helps in calculating centroids and updating on the dataset.
        Written By  : Atif Ali Mohammed
        """
        try:
            kmean_centroid = KMeans(n_clusters=n_clusters,random_state=42)
            kmean_centroid.fit_predict(df)
            centroids = kmean_centroid.cluster_centers_
            cen_x = [i[0] for i in centroids]
            cen_y = [i[1] for i in centroids]
            df['cen_x'] = df['cluster'].map(dict(enumerate(cen_x)))
            df['cen_y'] = df['cluster'].map(dict(enumerate(cen_y)))
            return df
        except OSError:
            with open('log/calculate_cluster.txt', 'a+') as file:
                self.log_writer.log(file,'Error Occurred while reducing dimensions of datasets :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open('log/calculate_cluster.txt', 'a+') as f:
                self.log_writer.log(f,'Error Occurred while rreducing dimensions of datasets :: {}'.format(e))
            raise e

    def updating_color_column(self,df,color):
        """
        Method Name : updating_color_column
        Description : It helps in updating color column to the dataset. So that each datapoint can be differentiated.
        Written By  : Atif Ali Mohammed
        """
        try:
            colors = ['b','g','r','k','c','m','y','b','g','r','k','c','m','y','b','g','r','k','c']
            df['color']=df['cluster'].map(dict(enumerate(colors)))
            return df
        except OSError:
            with open('log/calculate_cluster.txt', 'a+') as file:
                self.log_writer.log(file,'Error Occurred while reducing dimensions of datasets :: {}'.format(OSError))
            raise OSError
        except Exception as e:
            with open('log/calculate_cluster.txt', 'a+') as f:
                self.log_writer.log(f,'Error Occurred while rreducing dimensions of datasets :: {}'.format(e))
            raise e



cluster = CalculateCluster()
kmeans = cluster.forming_cluster(df=pca_transform,n_clusters=19)
pca_transform = cluster.calculate_centroid(df=pca_transform,n_clusters=19)
kmeans_df = cluster.updating_color_column(df=pca_transform,color=['b','g','r','k','c','m','y','b','g','r','k','c','m','y','b','g','r','k','c'])

pca_transform.to_csv('output.csv', index=False)   










        








