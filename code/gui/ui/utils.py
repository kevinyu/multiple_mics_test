from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import (
    pyqtSignal,
    Qt,
)

class QHLine(widgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(widgets.QFrame.HLine)
        self.setFrameShadow(widgets.QFrame.Sunken)


class QClickableLineEdit(widgets.QLineEdit):
    clicked = pyqtSignal() # signal when the text entry is left clicked

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: self.clicked.emit()
        else: super().mousePressEvent(event)
