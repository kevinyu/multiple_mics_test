# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'new_code/gui/ui/streamview.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_StreamView(object):
    def setupUi(self, StreamView):
        StreamView.setObjectName("StreamView")
        StreamView.resize(606, 213)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(StreamView)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.frame = QtWidgets.QFrame(StreamView)
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setContentsMargins(1, 1, 1, 1)
        self.gridLayout_2.setSpacing(1)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.indexLabel = QtWidgets.QLabel(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.indexLabel.sizePolicy().hasHeightForWidth())
        self.indexLabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.indexLabel.setFont(font)
        self.indexLabel.setObjectName("indexLabel")
        self.gridLayout_2.addWidget(self.indexLabel, 0, 0, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.amplitudeViewCheckBox = QtWidgets.QCheckBox(self.frame)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.amplitudeViewCheckBox.setFont(font)
        self.amplitudeViewCheckBox.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.amplitudeViewCheckBox.setChecked(True)
        self.amplitudeViewCheckBox.setObjectName("amplitudeViewCheckBox")
        self.horizontalLayout_4.addWidget(self.amplitudeViewCheckBox)
        self.spectrogramViewCheckBox = QtWidgets.QCheckBox(self.frame)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.spectrogramViewCheckBox.setFont(font)
        self.spectrogramViewCheckBox.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.spectrogramViewCheckBox.setChecked(True)
        self.spectrogramViewCheckBox.setObjectName("spectrogramViewCheckBox")
        self.horizontalLayout_4.addWidget(self.spectrogramViewCheckBox)
        self.gridLayout_2.addLayout(self.horizontalLayout_4, 0, 3, 1, 1)
        self.streamControls = QtWidgets.QWidget(self.frame)
        self.streamControls.setObjectName("streamControls")
        self.gridLayout = QtWidgets.QGridLayout(self.streamControls)
        self.gridLayout.setObjectName("gridLayout")
        self.channelLabel = QtWidgets.QLabel(self.streamControls)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.channelLabel.sizePolicy().hasHeightForWidth())
        self.channelLabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.channelLabel.setFont(font)
        self.channelLabel.setObjectName("channelLabel")
        self.gridLayout.addWidget(self.channelLabel, 0, 1, 1, 1)
        self.gainSpinner = QtWidgets.QDoubleSpinBox(self.streamControls)
        self.gainSpinner.setMinimumSize(QtCore.QSize(50, 0))
        self.gainSpinner.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.gainSpinner.setDecimals(1)
        self.gainSpinner.setMinimum(-99.0)
        self.gainSpinner.setMaximum(99.0)
        self.gainSpinner.setObjectName("gainSpinner")
        self.gridLayout.addWidget(self.gainSpinner, 1, 0, 1, 1)
        self.thresholdSpinner = QtWidgets.QSpinBox(self.streamControls)
        self.thresholdSpinner.setMinimumSize(QtCore.QSize(50, 0))
        self.thresholdSpinner.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.thresholdSpinner.setMaximum(9999)
        self.thresholdSpinner.setSingleStep(20)
        self.thresholdSpinner.setObjectName("thresholdSpinner")
        self.gridLayout.addWidget(self.thresholdSpinner, 3, 0, 1, 1)
        self.gainLabel = QtWidgets.QLabel(self.streamControls)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gainLabel.sizePolicy().hasHeightForWidth())
        self.gainLabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.gainLabel.setFont(font)
        self.gainLabel.setObjectName("gainLabel")
        self.gridLayout.addWidget(self.gainLabel, 0, 0, 1, 1)
        self.channelDropdown = QtWidgets.QComboBox(self.streamControls)
        self.channelDropdown.setMaximumSize(QtCore.QSize(50, 16777215))
        self.channelDropdown.setObjectName("channelDropdown")
        self.gridLayout.addWidget(self.channelDropdown, 1, 1, 1, 1)
        self.thresholdSlider = QtWidgets.QSlider(self.streamControls)
        self.thresholdSlider.setMinimumSize(QtCore.QSize(0, 0))
        self.thresholdSlider.setMaximumSize(QtCore.QSize(50, 16777215))
        self.thresholdSlider.setMinimum(0)
        self.thresholdSlider.setMaximum(9999)
        self.thresholdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.thresholdSlider.setObjectName("thresholdSlider")
        self.gridLayout.addWidget(self.thresholdSlider, 3, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 4, 0, 1, 1)
        self.thresholdLabel = QtWidgets.QLabel(self.streamControls)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.thresholdLabel.setFont(font)
        self.thresholdLabel.setObjectName("thresholdLabel")
        self.gridLayout.addWidget(self.thresholdLabel, 2, 0, 1, 1)
        self.gridLayout_2.addWidget(self.streamControls, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.triggeredButton = QtWidgets.QPushButton(self.frame)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.triggeredButton.setFont(font)
        self.triggeredButton.setCheckable(True)
        self.triggeredButton.setChecked(True)
        self.triggeredButton.setObjectName("triggeredButton")
        self.triggeredButtons = QtWidgets.QButtonGroup(StreamView)
        self.triggeredButtons.setObjectName("triggeredButtons")
        self.triggeredButtons.addButton(self.triggeredButton)
        self.horizontalLayout_2.addWidget(self.triggeredButton)
        self.continuousButton = QtWidgets.QPushButton(self.frame)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.continuousButton.setFont(font)
        self.continuousButton.setCheckable(True)
        self.continuousButton.setChecked(False)
        self.continuousButton.setObjectName("continuousButton")
        self.triggeredButtons.addButton(self.continuousButton)
        self.horizontalLayout_2.addWidget(self.continuousButton)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 2, 2, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.recordButton = QtWidgets.QPushButton(self.frame)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.recordButton.setFont(font)
        self.recordButton.setStyleSheet("")
        self.recordButton.setCheckable(True)
        self.recordButton.setObjectName("recordButton")
        self.recordButtons = QtWidgets.QButtonGroup(StreamView)
        self.recordButtons.setObjectName("recordButtons")
        self.recordButtons.addButton(self.recordButton)
        self.horizontalLayout.addWidget(self.recordButton)
        self.monitorButton = QtWidgets.QPushButton(self.frame)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.monitorButton.setFont(font)
        self.monitorButton.setCheckable(True)
        self.monitorButton.setChecked(True)
        self.monitorButton.setObjectName("monitorButton")
        self.recordButtons.addButton(self.monitorButton)
        self.horizontalLayout.addWidget(self.monitorButton)
        self.gridLayout_2.addLayout(self.horizontalLayout, 2, 3, 1, 1)
        self.plotView = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plotView.sizePolicy().hasHeightForWidth())
        self.plotView.setSizePolicy(sizePolicy)
        self.plotView.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.plotView.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.plotView.setObjectName("plotView")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.plotView)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.plotViewLayout = QtWidgets.QVBoxLayout()
        self.plotViewLayout.setSpacing(1)
        self.plotViewLayout.setObjectName("plotViewLayout")
        self.verticalLayout_2.addLayout(self.plotViewLayout)
        self.gridLayout_2.addWidget(self.plotView, 1, 1, 1, 3)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.streamNameLabel = QtWidgets.QLabel(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.streamNameLabel.sizePolicy().hasHeightForWidth())
        self.streamNameLabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.streamNameLabel.setFont(font)
        self.streamNameLabel.setObjectName("streamNameLabel")
        self.horizontalLayout_5.addWidget(self.streamNameLabel)
        self.editNameButton = QtWidgets.QPushButton(self.frame)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setUnderline(False)
        font.setStrikeOut(False)
        self.editNameButton.setFont(font)
        self.editNameButton.setAutoFillBackground(False)
        self.editNameButton.setStyleSheet("QPushButton {\n"
"background: none;\n"
"color: rgb(16, 37, 127);\n"
"text-align: left;\n"
"border: none;\n"
"text-decoration: none;\n"
"}\n"
"     \n"
"QPushButton:hover {\n"
"color: rgb(116, 137, 127);\n"
"text-decoration: underline;\n"
"}")
        self.editNameButton.setObjectName("editNameButton")
        self.horizontalLayout_5.addWidget(self.editNameButton)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.gridLayout_2.addLayout(self.horizontalLayout_5, 0, 1, 1, 2)
        self.detectionIndicatorLabel = QtWidgets.QLabel(self.frame)
        self.detectionIndicatorLabel.setObjectName("detectionIndicatorLabel")
        self.gridLayout_2.addWidget(self.detectionIndicatorLabel, 2, 0, 1, 2)
        self.horizontalLayout_3.addWidget(self.frame)

        self.retranslateUi(StreamView)
        QtCore.QMetaObject.connectSlotsByName(StreamView)

    def retranslateUi(self, StreamView):
        _translate = QtCore.QCoreApplication.translate
        StreamView.setWindowTitle(_translate("StreamView", "Form"))
        self.indexLabel.setText(_translate("StreamView", "1"))
        self.amplitudeViewCheckBox.setText(_translate("StreamView", "Amplitude"))
        self.spectrogramViewCheckBox.setText(_translate("StreamView", "Spectrogram"))
        self.channelLabel.setText(_translate("StreamView", "Channel"))
        self.gainLabel.setText(_translate("StreamView", "Gain"))
        self.thresholdLabel.setText(_translate("StreamView", "Thresh"))
        self.triggeredButton.setText(_translate("StreamView", "Triggered"))
        self.continuousButton.setText(_translate("StreamView", "Continuous"))
        self.recordButton.setText(_translate("StreamView", "Record"))
        self.monitorButton.setText(_translate("StreamView", "Monitor"))
        self.streamNameLabel.setText(_translate("StreamView", "Stream Name"))
        self.editNameButton.setText(_translate("StreamView", "edit"))
        self.detectionIndicatorLabel.setText(_translate("StreamView", "-"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    StreamView = QtWidgets.QWidget()
    ui = Ui_StreamView()
    ui.setupUi(StreamView)
    StreamView.show()
    sys.exit(app.exec_())