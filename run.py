from nupic.frameworks.opf.model_factory import ModelFactory
import json
import numpy as np
import matplotlib.pyplot as plt
from nupic.algorithms.anomaly_likelihood import AnomalyLikelihood

def get_params():
    parameters = open('parameters.json',)
    parameters = json.load(parameters)
    return parameters

def create_model(parameters):
    model = ModelFactory.create(modelConfig = parameters["modelConfig"])
    model.enableLearning()
    model.enableInference(parameters["inferenceArgs"])
    return model

def run_model(model, a, b, save = True):
    signal = np.load("./signs/sign.npy")
    signal = signal[a:b,1]

    anom_scores = []
    anom_likelihood = []
    anom_loglikelihood =[]

    likelihood = AnomalyLikelihood(learningPeriod = 300)

    for value in signal:
        inputRecords={}
        inputRecords['c1'] = float(value)

        result = model.run(inputRecords)

        current_likelihood = likelihood.anomalyProbability(value, result.inferences["anomalyScore"], timestamp = None)
        current_loglikelihood = likelihood.computeLogLikelihood(current_likelihood)


        anom_scores.append(result.inferences["anomalyScore"])
        anom_likelihood.append(current_likelihood)
        anom_loglikelihood.append(current_loglikelihood)
    
    if save == True:
        np.savetxt("anom_score.txt",anom_scores,delimiter=',')

        np.savetxt("anom_likelihood.txt",anom_likelihood,delimiter=',')

        np.savetxt("anom_logscore.txt", anom_loglikelihood,delimiter=',')


def plot():

    anom_scores = np.genfromtxt("anom_score.txt", delimiter = ",")
    anom_scores = [anom_scores[i] for i in range(np.size(anom_scores)) if i%2==0]
 

    anom_loglikelihood = np.genfromtxt("anom_logscore.txt", delimiter = ",")

    plt.plot(np.arange(np.size(anom_scores)), anom_scores)
    plt.plot(np.arange(np.size(anom_scores)), anom_loglikelihood, color = 'red')
    plt.show()


def main():
    PARAMS_ = get_params()
    model = create_model(PARAMS_)
    a, b = (7160000, 7180000)
    run_model(model, a, b, save = True)
    plot()



if __name__ == '__main__':
    main()