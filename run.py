# -*- coding: utf-8 -*-
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
    Arguments:
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
    for i, v in enumerate(signs[:,1]):                                  # iterates over the signs. The "v" isn't usefull.

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

    ##################### declare the predicted list ###################################
    predictions_1 = []
    predictions_5 = []
    predictions_1.append(0)
    for i in range(5):
        predictions_5.append(0)                                             # as this prediction is always made 1 step ahead, then the first value predicted will be ...
                                                                            # the prediction of the index with number 1, therefore doesn't exist a prediction of the 0 ...
                                                                            # index. The same problem occurs with the last signal, because it will predict one more ...
                                                                            # step ahead, this means that after seen the last signal "A", it will predict "A+1" even it doesnt ...
                                                                            # having a matching value in the signal array.
    #-----------------------------------------------------------------------------------

    ################ declare the Anom likelihood class #################################
    likelihood = AnomalyLikelihood(learningPeriod = 300)
    #-----------------------------------------------------------------------------------


    for counter, value in enumerate(signal):                                        # iterate over each value in the signal array, the  counter is used for debugging purposes
        
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
        ################################ PREDICTIONS ####################################
        bestPredictions = result.inferences["multiStepBestPredictions"]                                      # obtain the predicted value from infereces dict
        predictions_1.append(bestPredictions[1])
        predictions_5.append(bestPredictions[5])                                         # append the value to the _predict array   
        
        #--------------------------------------------------------------------------------


        ########### add the anomaly values to the respective list #######################
        anom_scores.append(result.inferences["anomalyScore"])
        anom_likelihood.append(current_likelihood)
        anom_loglikelihood.append(current_loglikelihood)
        #--------------------------------------------------------------------------------
        ################# print the input and prediction, for debugging purposes ########
        if counter % 1 == 0:
            #print("Actual input [%d]: %f" % (counter, value))
            print('prediction of [{0}]:(input) {1:8} (1-step) {2:8} (5-step) {3:8}'.format(counter, value, predictions_1[counter], predictions_5[counter]))
            #print("Input[%d]: %f" % (counter+1,signal[counter+1]))
            #print("Multi Step Predictions: %s" % (result.inferences["multiStepPredictions"]))
            #print("\n")
        #--------------------------------------------------------------------------------

    ################# save the anomaly and prediction array #########################
    if save == True:
        np.savetxt("anom_score_"+ string + ".txt",anom_scores,delimiter=',')    # the "string" is to differentiate the training and ...
                                                                                # the online learning outputs.

        np.savetxt("anom_likelihood_" + string + ".txt",anom_likelihood,delimiter = ',')

        np.savetxt("anom_logscore_" + string + ".txt", anom_loglikelihood, delimiter = ',')

        np.savetxt("anom_prediction_1" + string + ".txt", predictions_1, delimiter = ',')

        np.savetxt("anom_prediction_5" + string + ".txt", predictions_5, delimiter = ',')
    #--------------------------------------------------------------------------------

def plot(a,b,  aggregate = False, string = ''):
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

    ################# open the anom score, logscore and predictions files ##########################
    anom_scores = np.genfromtxt("anom_score_"+ string + ".txt", delimiter = ",")
    anom_loglikelihood = np.genfromtxt("anom_logscore_" + string + ".txt", delimiter = ",")
    prediction_1 = np.genfromtxt("anom_prediction_1" + string + ".txt", delimiter = ",")
    prediction_5 = np.genfromtxt("anom_prediction_5" + string + ".txt", delimiter = ",")
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


    ################# plot the data and the prediction #####################################

    if np.size(prediction_1)-1 == np.size(anom_scores) and np.size(prediction_5)-5 == np.size(anom_scores):         # testando o tamanho dos arrays. Por lógica...
        print("o tamanho do array de predicao-1 e de anomalias eh igual")                                           # o primeiro elemento dos sinais (sign[0]) não possui uma predição - o classifier ...
                                                                                                                    # prediz, com base no sign[0] o sign[1]. Porém, o classifier também ...
                                                                                                                    # prediz um elemento a mais depois do final dos sinais. Por isso adiciono um "Nan" ...
                                                                                                                    # como primeiro elemento do array _predictions e também, na hora de plotar ...
                                                                                                                    # limito o array _predictions para que não utilize a última predição, pois não há ...
                                                                                                                    # input que corresponda a ela.


    plt.plot(np.arange(np.size(anom_scores)), scalar_1, "s", color = "black")
    plt.plot(np.arange(np.size(prediction_1)-1), prediction_1[:-1],'v', color = 'red')
    plt.plot(np.arange(np.size(prediction_5)-5), prediction_5[:-5],'o', color = 'blue')
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

    PARAMS_ = get_params()                                                                              # get the params
    model = create_model(PARAMS_)                                                                       # creates the model

    #c, d = (7090000, 7100000)                                                                          # define the index of the values which the model will be trained.
    #model = training_(model, c, d, save = True)                                                        # train the model 

    #model.save('C:/Users/Usuario/Desktop/HTM/OPF/saveModel_classifier')                                # save the model
    #model = ModelFactory.loadFromCheckpoint('C:/Users/Usuario/Desktop/HTM/OPF/saveModel_classifier')   # load the model

    a, b = (14500000, 14505000)                                                                         # define the index of the values which will be ran over by the HTM

    #most_used_string = ['after_training', 'training_1','training_2','after_load', 'classifier_training']
    #run_model(model, a, b, save = True, aggregate = False, string='classifier_training')                # run the model

    plot( a, b, aggregate = False, string = 'classifier_training')                                      # plot the data


if __name__ == '__main__':
    main()