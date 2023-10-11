import numpy as np

def MD(self, processed_data, training, test, args=[1]):

    try:
        z_MD = args[0]

        covariance  = np.cov(processed_data.loc[training,:].dropna(), rowvar=False)
        cov_inv = np.linalg.matrix_power(covariance, -1)
        centerpoint = np.mean(processed_data.loc[training,:].dropna() , axis=0)
        
        distances = []

        for ind in processed_data.loc[test,:].index:
            if processed_data.loc[ind,:].isnull().values.any():
                distances.append(np.nan)
            else:
                p1 = processed_data.loc[ind,:]
                p2 = centerpoint
                distance = (p1-p2).T.dot(cov_inv).dot(p1-p2)
                distances.append(distance)

        distances = np.array(distances)
                
        # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers 
        #cutoff = chi2.ppf(confidence, processed_data.loc[training,:].dropna().shape[1])

        #print(np.where(distances > cutoff )[0])

        #outlierIndexes = np.array(np.where(distances > cutoff )[0])

        
        return np.sqrt(distances) / z_MD
    
    except Exception as e:
        print("An exception occurred during MD:" + str(e))
        return np.repeat(np.nan, len(test))           

    