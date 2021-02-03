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

def open_signs():
    """Return the signs
    """
    return np.load("./signs/sign.npy")

def run_model(model, a, b, save = True):
    """Runs the HTM model and generates the anomaly scores"
    :model: the model created with create_model()
    :a: the beginning of the anylized signal
    :b: the end of the anylized signal
    :save: if True then the anomalies output will be saved as .txt
    """

    ######################### open the signs ###########################################
    signal = open_signs()
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
        np.savetxt("anom_score_2.txt",anom_scores,delimiter=',')

        np.savetxt("anom_likelihood_2.txt",anom_likelihood,delimiter=',')

        np.savetxt("anom_logscore_2.txt", anom_loglikelihood,delimiter=',')
    #--------------------------------------------------------------------------------

def plot(a,b):
    """Plots the anomaly score and likelihood in the same window. 
    """
    ############## open the sign and vector time .txt files ##########################
    scalar_1 = open_signs()[a:b,1]
    time_vect = open_signs()[a:b,0]
    #---------------------------------------------------------------------------------

    ################# open the anom score and logscore .txt ##########################
    anom_scores = np.genfromtxt("anom_score.txt", delimiter = ",")
    anom_loglikelihood = np.genfromtxt("anom_logscore.txt", delimiter = ",")
    #---------------------------------------------------------------------------------
    
    ############ plot the anomaly likelihood and the signal in the same plot #########
    fig, axs = plt.subplots(2)                    # declare the plot 

    axs[0].set_ylabel("y position")
    axs[0].plot(time_vect,scalar_1,'o',color="black")

    axs[1].set_ylabel("anomaly loglikelihood")
    axs[1].plot(time_vect, anom_loglikelihood, color = "blue")
    axs[1].set_xlabel("time(ms)")

    plt.show()
    #--------------------------------------------------------------------------------

    ################# plot both anomalies types #####################################
    plt.plot(np.arange(np.size(anom_scores)), anom_scores)
    plt.plot(np.arange(np.size(anom_scores)), anom_loglikelihood, color = 'red')
    plt.show()
    #--------------------------------------------------------------------------------

def main():
    #PARAMS_ = get_params() # get the params
    #model = create_model(PARAMS_) # creates the model
    a, b = (7160000, 7300000) # define the index of the values which will be ran over by the HTM
    #run_model(model, a, b, save = True) # run the model
    plot(a,b) # plot the data



if __name__ == '__main__':
    main()