import sys
from PyQt5 import QtWidgets
from app import App, AppOptions


def main():
    q_app = QtWidgets.QApplication(sys.argv)
    app = App()

    # option = int(input(app.getAppInstructions()))
    option = AppOptions.viewData.value
    if option == AppOptions.collectData.value:
        app.collectTrainingData()
        input('Press Enter when you are done')
        app.endDataCollection()
        app.displaySampleFromCollectedData()
    if option == AppOptions.trainModel.value:
        app.trainModel()
    if option == AppOptions.predict.value:
        app.predictData()
        input('Press Enter when you are done')
    if option == AppOptions.viewData.value:
        app.viewData()

    sys.exit(q_app.exec())


if __name__ == '__main__':
    main()
