from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
import mainWindow

class PlotsApp(QtWidgets.QMainWindow, mainWindow.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self) 


app = QApplication([])
window = PlotsApp()
window.show()
app.exec_()