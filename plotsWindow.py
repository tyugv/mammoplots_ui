from PyQt5 import QtWidgets

class Ui_PlotsWindow(object):

    def setupUi(self, MainWindow, sin1, sin2):

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 700)

        self.sin1 = QtWidgets.QLabel(self)
        sin1 = sin1.scaled(350, 350)
        self.sin1.setPixmap(sin1)
        self.sin1.resize(sin1.size())
        self.sin1.move(10, 10)

        self.sin2 = QtWidgets.QLabel(self)
        sin2 = sin2.scaled(350, 350)
        self.sin2.setPixmap(sin2)
        self.sin2.resize(sin2.size())
        self.sin2.move(500, 10)

class PlotsWindow(QtWidgets.QMainWindow, Ui_PlotsWindow):
    def __init__(self, sin1, sin2):
        super().__init__()
        self.setupUi(self, sin1 = sin1, sin2 = sin2) 

