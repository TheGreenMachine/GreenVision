import networktables as nt
import time
import sys

values = {}
values['visionX'] = -99
values['visionY'] = -99
values['width'] = -99
values['height'] = -99
values['center0_x'] = -99
values['center0_y'] = -99
values['center1_x'] = -99
values['center1_y'] = -99
values['center_x'] = -99
values['center_y'] = -99
values['distance_esti'] = -99


def view():
    #global values
    msg = "\rvis: {:3.1f} {:3.1f}; wid,ht: {:3.1f} {:3.1f}; cent0: {:3.1f} {:3.1f}; cent1: {:3.1f} {:3.1f}, cent: {:3.1f} {:3.1f}; dist: {:3.1f}            ".format( values['visionX'], values['visionY'], values['width'], values['height'], values['center0_x'], values['center0_y'], values['center1_x'], values['center1_y'], values['center_x'], values['center_y'], values['distance_esti'])
    sys.stdout.write(msg)


def value_changed(table, key, value, isNew):
    global values
    values[key] = value
    if key == 'distance_esti':
        view()

def start_listener():
    table.addEntryListener(value_changed, key="visionX")
    table.addEntryListener(value_changed, key="visionY")
    table.addEntryListener(value_changed, key="width")
    table.addEntryListener(value_changed, key="height")

    table.addEntryListener(value_changed, key="center0_x")
    table.addEntryListener(value_changed, key="center0_y")
    table.addEntryListener(value_changed, key="center1_x")
    table.addEntryListener(value_changed, key="center1_y")
    table.addEntryListener(value_changed, key="center_x")
    table.addEntryListener(value_changed, key="center_y")
    table.addEntryListener(value_changed, key="distance_esti")

nt.NetworkTables.initialize(server='10.18.16.2')
table = nt.NetworkTables.getTable('SmartDashboard')

view()

start_listener()

while True:
    time.sleep(1)
