from nupic.frameworks.opf.common_models.cluster_params import getScalarMetricWithTimeOfDayAnomalyParams

def get_params(min_val, max_val):
    """
    Returns a dict containing the model parameters. 
    :min_val: the 'expected' minimum value of the scalar data
    :max_val: the 'expected' max value of the scalar data
    """
    params = getScalarMetricWithTimeOfDayAnomalyParams(
        metricData=[0],
        tmImplementation="cpp", 
        minVal = min_val, 
        maxVal = max_val)  
    
    return params

