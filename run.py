from nupic.frameworks.opf.model_factory import ModelFactory
import json
import numpy as np
import matplotlib.pyplot as plt
from nupic.algorithms.anomaly_likelihood import AnomalyLikelihood

def get_params():
    """Opens the .json params file and returns it. 
    """
    parameters = open('parameters.json',) # open the .json parameters
    parameters = json.load(parameters)
    return parameters

def create_model(parameters):
    """Creates the HTM model and returns it.
    :parameters: the .json parameters created with params.py and opened by create_model()
    """

    model = ModelFactory.create(modelConfig = parameters["modelConfig"]) # creates the HTM model with...
    # the parameters["modelConfig"] parameters
    model.enableLearning() # enable the learning of the model
    model.enableInference(parameters["inferenceArgs"]) # tells the model to Infer about the specified informations...
    # in parameters["inferenceArgs"] - for example, it can tell the HTM model to calculate the anomaly score or run ...
    # the classifier. 
    return model

def run_model(model, a, b, save = True):
    """Runs the HTM model and generates the anomaly scores"
    :model: the model created with create_model()
    :a: the beginning of the anylized signal
    :b: the end of the anylized signal
    :save: if True then the anomalies output will be saved as .txt
    """

    ######################### open the signs ###########################################
    signal = np.load("./signs/sign.npy")
    signal = signal[a:b,1]
    #-----------------------------------------------------------------------------------

    ##################### declare the anomalies lists ##################################
    anom_scores = []
    anom_likelihood = []
    anom_loglikelihood =[]
    #-----------------------------------------------------------------------------------

    ################ declare the Anom likelihood class #################################
    likelihood = AnomalyLikelihood(learningPeriod = 300)
    #-----------------------------------------------------------------------------------


    for value in signal:# iterate over each value in the signal array
        
        ############ declare the dict which will be passed to the model ##############
        inputRecords={}                   # the model only accepts data in a specific dict format ...                
        inputRecords['c1'] = float(value) # this format is shown here: 
        #-------------------------------------------------------------------------------

        ############ run the HTM model over the inputRecords dict ######################
        result = model.run(inputRecords)
        #-------------------------------------------------------------------------------

        ############ compute the anomaly likelihood and loglikelihood ###################
        current_likelihood = likelihood.anomalyProbability(value, result.inferences["anomalyScore"], timestamp = None)
        current_loglikelihood = likelihood.computeLogLikelihood(current_likelihood)
        #--------------------------------------------------------------------------------

        ########### add the anomaly values to the respective list #######################
        anom_scores.append(result.inferences["anomalyScore"])
        anom_likelihood.append(current_likelihood)
        anom_loglikelihood.append(current_loglikelihood)
        #--------------------------------------------------------------------------------

    ################# save the anomaly values #######################################
    if save == True:
        np.savetxt("anom_score.txt",anom_scores,delimiter=',')

        np.savetxt("anom_likelihood.txt",anom_likelihood,delimiter=',')

        np.savetxt("anom_logscore.txt", anom_loglikelihood,delimiter=',')
    #--------------------------------------------------------------------------------

def plot():
    """Plots the anomaly score and likelihood in the same window. 
    """
    ################# open the anom score and logscore .txt ##########################
    anom_scores = np.genfromtxt("anom_score.txt", delimiter = ",")
    anom_loglikelihood = np.genfromtxt("anom_logscore.txt", delimiter = ",")
    #--------------------------------------------------------------------------------

    ################# plot both anomalies types #####################################
    plt.plot(np.arange(np.size(anom_scores)), anom_scores)
    plt.plot(np.arange(np.size(anom_scores)), anom_loglikelihood, color = 'red')
    plt.show()
    #--------------------------------------------------------------------------------

def main():
    PARAMS_ = get_params() # get the params
    model = create_model(PARAMS_) # creates the model
    a, b = (7160000, 7162000) # define the index of the values which will be ran over by the HTM
    run_model(model, a, b, save = True) # run the model
    plot() # plot the data



if __name__ == '__main__':
    main()