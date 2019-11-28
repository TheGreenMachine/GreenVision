from GreenVision import GreenVision
gv = GreenVision("settings.yaml")
gv.initNetworkTables()
while True:
    gv.capturePhotoAndFilter()
    gv.findContours()
    gv.genRectangleList()
    gv.checkCenterCords()
    gv.showWindow()