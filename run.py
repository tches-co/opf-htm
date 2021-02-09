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

    model = ModelFactory.create(modelConfig = parameters["modelConfig"])        # creates the HTM model with the parameters["modelConfig"] parameters
    model.enableLearning()                                                      # enable the learning of the model
    model.enableInference(parameters["inferenceArgs"])                          # tells the model to Infer about the specified informations...
                                                                                # in parameters["inferenceArgs"] - for example, it can tell the HTM model ...
                                                                                # to calculate the anomaly score or run the classifier.  
    return model


def aggregate_(a,b):
    """
    Aggregate the data by doing the mean for each indice values divisible by 3 and the next 2 values.
    Arguments:
        :a: the beginning of the anylized signal.
        :b: the end of the anylized signal.
    It returns:
        :scalar_1: the c1 values list
        :time_vect: the time vector list
    """
    signs = open_signs()[a:b,0:2]

    scalar_1  = []
    time_vect = []
    for i, v in enumerate(signs[:,1]): #iterates over the signs

        try:                                                            # firstly we try iterating over the whole data doing the mean for the next 2 positions every i.
            if i%3 == 0:                                                # take 1 every 3 positions and then take and average of i and the 2 subsequent values.
                scalar = (signs[i,1] + signs[i+1,1] + signs[i+2,1])/3   # taking the average of the i and the next 2 values
                time_ = signs[i,0]                                      # saving each i "time" to make the plot easier. Add the values, we pick 1 every 3 datapoints.

                scalar_1.append(scalar)                                 # we append those values in each array
                time_vect.append(time_)   

        except IndexError:                                              # then, if we can't reach the i+2 index, then we try to reach only the i+1 value
            try: 
                scalar = (signs[i,1] + signs[i+1,1])/2
                time_ = signs[i,0]

                scalar_1.append(scalar)
                time_vect.append(time_)
                

            except IndexError:                                          # then, if we can't reach the i+1 index, we only use the i itself - neither the i+1 nor the i+2
                scalar = signs[i,1] 
                time_ = signs[i,0]

                scalar_1.append(scalar)
                time_vect.append(time_)
            
    return scalar_1, time_vect


def open_signs():
    """Return the signs
    """
    return np.load("./signs/sign.npy")

def run_model(model, a, b, save = True, aggregate = False, string = ''):
    """Runs the HTM model and generates the anomaly scores.
    Arguments:
        :model: the model created with create_model().
        :a: the beginning of the anylized signal.
        :b: the end of the anylized signal.
        :save: if True then the anomalies output will be saved as .txt.
        :string: the string to differentiate the name of the saved .txt files.
    """

    ######################### open the signs ###########################################
    if aggregate == True:
        signal, time_vect = aggregate_(a,b)
        print("the size of signal is: {i}".format(i=np.size(signal)))

    else:
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


    for value in signal:                                                            # iterate over each value in the signal array
        
        ############ declare the dict which will be passed to the model ###############
        inputRecords={}                                                             # the model only accepts data in a specific dict format ...                
        inputRecords['c1'] = float(value)                                           # this format is shown here: 
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
        np.savetxt("anom_score_"+ string + ".txt",anom_scores,delimiter=',')    # the "string" is to differentiate the training and ...
                                                                                # the online learning outputs.

        np.savetxt("anom_likelihood_" + string + ".txt",anom_likelihood,delimiter=',')

        np.savetxt("anom_logscore_" + string + ".txt", anom_loglikelihood,delimiter=',')
    #--------------------------------------------------------------------------------

def plot(a,b,  aggregate = False):
    """Plots the anomaly score and likelihood in the same window. 
    Arguments:

        :a: the beginning of the plot.
        :b: the end of the plot.
        :aggregate: if True, the data will be aggregated. 
    """
    ############## open the sign and vector time .txt files ##########################
    if aggregate == True:                          
        scalar_1, time_vect = aggregate_(a,b)           # aggregate the data, get the scalar and the temporal part of signs
    else:
        scalar_1 = open_signs()[a:b,1]                  # get the scalar parte of the signs
        time_vect = open_signs()[a:b,0]                 # the the "temporal" parte of the signs
    #---------------------------------------------------------------------------------

    ################# open the anom score and logscore .txt ##########################
    anom_scores = np.genfromtxt("anom_score_after_training.txt", delimiter = ",")
    anom_loglikelihood = np.genfromtxt("anom_logscore_after_training.txt", delimiter = ",")
    #---------------------------------------------------------------------------------
    
    ############ plot the anomaly likelihood and the signal in the same plot #########
    fig, axs = plt.subplots(2)                          # declare the figure with 2 plots 

    axs[0].set_ylabel("y position")             
    axs[0].plot(time_vect,scalar_1,'--bo',color="black")

    axs[1].set_ylabel("anomaly score")
    axs[1].plot(time_vect, anom_scores, color = "blue")
    axs[1].set_xlabel("time(ms)")

    plt.show()
    #--------------------------------------------------------------------------------

    ################# plot both anomalies types #####################################
    plt.plot(np.arange(np.size(anom_scores)), anom_scores)
    plt.plot(np.arange(np.size(anom_scores)), anom_loglikelihood, color = 'red')
    plt.show()
    #--------------------------------------------------------------------------------

def training_(model, c, d, save = False):
    """
    Train the model on the [c:d] region of the dataset.
    Arguments:
        :model: the HTM model.
        :c: the beginning of the training .
        :d: the end of the training.
        :save: if True, then the training anomaly scores will be saved.
    Returns:
        :model: the HTM model.
    """
    training_1, time_vect_training = aggregate_(c,d) # get the aggregated data for training
    
    n = 10                                           # number of batches on the training data

    ################## the model iterates over the data n times #####################
    for count in range(n): 
        txt = "training_" + "{i}".format(i=count) 
        run_model(model, c, d, save= False, aggregate= True, string = txt)
    #--------------------------------------------------------------------------------

    return model


def main():

    PARAMS_ = get_params()                                                          # get the params
    model = create_model(PARAMS_)                                                   # creates the model

    c, d = (7090000, 7100000)                                                       # define the index of the values which the model will be trained.
    model = training_(model, c, d, save = False)                                    # train the model 

    a, b = (7090000, 7400000)                                                       # define the index of the values which will be ran over by the HTM
    model.save('C:/Users/Usuario/Desktop/HTM/OPF/saveModel')                        # save the model
    run_model(model, a, b, save = True, aggregate = True, string='after_training')  # run the model

    plot(a,b, aggregate = True)                                                     # plot the data




if __name__ == '__main__':
    main()