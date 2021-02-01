from nupic.frameworks.opf.common_models.cluster_params import getScalarMetricWithTimeOfDayAnomalyParams
import pprint
import yaml
import json


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
    
    with open('parameters.json','w') as outfile:
        json.dump(params, outfile, indent = 4)
    return params

def main():
    pp = pprint.PrettyPrinter(indent = 1)
    pp.pprint(get_params(-100,100))

if __name__ == "__main__":
    main()