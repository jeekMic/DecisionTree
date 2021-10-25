import sys

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt

from DecisionTree import SoilPrediction


class Winform(QWidget):
    def __init__(self, parent=None):
        super(Winform, self).__init__(parent)
        self.setWindowTitle("标哥的决策树算法")
        self.setFixedSize(1800, 600)

        hlayout = QVBoxLayout()

        # 水平居左 垂直居上
        self.lab1 = QLabel(self)
        self.lab1.setFixedSize(1400, 400)
        self.lab1.setScaledContents(True)
        self.lab1.setPixmap(QPixmap("demo.png"))

        hlayout.addWidget(self.lab1)


        self.setLayout(hlayout)


if __name__ == "__main__":
    model = SoilPrediction("./data2012to2013.xls")
    model.readData()
    model.dataProcessing()
    model.dataSetPartitioning()
    model.featureEngineering()
    model.decisionTree()
    model.drawTree()

    app = QApplication(sys.argv)
    form = Winform()
    form.show()
    sys.exit(app.exec_())
