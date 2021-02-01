from nupic.frameworks.opf.model_factory import ModelFactory
import json
import numpy as np
import matplotlib.pyplot as plt

def get_params():
    parameters = open('parameters.json',)
    parameters = json.load(parameters)
    return parameters

def create_model(parameters):
    model = ModelFactory.create(modelConfig = parameters["modelConfig"])
    model.enableLearning()
    model.enableInference(parameters["inferenceArgs"])
    return model

def run_model(model, a, b):
    signal = np.load("./signs/sign.npy")
    signal = signal[a:b,1]
    anom_scores = []

    for value in signal:
        inputRecords={}
        inputRecords['c1'] = float(value)
        result = model.run(inputRecords)
        anom_scores.append(result.inferences["anomalyScore"])
    
    plt.plot(np.arange(np.size(anom_scores)), anom_scores)
    plt.show()




def main():
    PARAMS_ = get_params()
    model = create_model(PARAMS_)
    a, b = (7160000, 7162000)
    run_model(model, a, b)


if __name__ == '__main__':
    main()