import enum
from training_data_collecter import TrainingDataCollector


class AppOptions(enum.Enum):
    collectData = 1
    predict = 2


class App:
    def __init__(self):
        self.trainingDataCollector = TrainingDataCollector()

    def getAppInstructions(self):
        instructions = "The following options are available:\n"
        for index, option in enumerate(AppOptions):
            instructions += str(index) + ': ' + str(option) + "\n"
        instructions += "Enter your choice: "
        return instructions

    # def processUserOption(self, option):
    #     print (f'Got option: {option}')
    #     if option == AppOptions.collectData:
    #         self.collectTrainingData()
    #     if option == AppOptions.predict:
    #         self.predictData()

    def predictData(self):
        print('Predicting data...')

    def collectTrainingData(self):
        self.trainingDataCollector.startCollecting()

    def endDataCollection(self):
        self.trainingDataCollector.endDataCollection()

    def displaySampleFromCollectedData(self):
        self.trainingDataCollector.displaySampleFromCollectedData()
