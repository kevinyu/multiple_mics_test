from PyQt5 import QtWidgets as widgets


class QHLine(widgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(widgets.QFrame.HLine)
        self.setFrameShadow(widgets.QFrame.Sunken)
