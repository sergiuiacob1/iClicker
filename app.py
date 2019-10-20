from training_data_collecter import TrainingDataCollector


class App:
    def __init__(self):
        self.trainingDataCollector = TrainingDataCollector()

    def collectTrainingData(self):
        self.trainingDataCollector.startCollecting()
