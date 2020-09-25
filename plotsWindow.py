from PyQt5 import QtWidgets

class Ui_PlotsWindow(object):

    def setupUi(self, MainWindow, img1, img2):

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 700)

        self.img1 = QtWidgets.QLabel(self)
        #sin1 = sin1.scaled(350, 350)
        self.img1.setPixmap(img1)
        self.img1.resize(img1.size())
        self.img1.move(10, 10)

        self.img2 = QtWidgets.QLabel(self)
        #sin2 = sin2.scaled(350, 350)
        self.img2.setPixmap(img2)
        self.img2.resize(img2.size())
        self.img2.move(500, 10)

class PlotsWindow(QtWidgets.QMainWindow, Ui_PlotsWindow):
    def __init__(self, img1, img2):
        super().__init__()
        self.setupUi(self, img1 = img1, img2 = img2) 

