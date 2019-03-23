import networktables as nt
import time
import sys
import csv

values = {}
values['visionX'] = -99
values['visionY'] = -99
values['width'] = -99
values['height'] = -99
values['center_x'] = -99
values['center_y'] = -99
values['distance_esti'] = -99
values['contours'] = -99
values['targets'] = -99
values['yaw'] = -99
values['pitch'] = -99


def view():
    # global values
    msg = "\rw, h: ({} {});" \
          " coords: ({} {});" \
          " yaw, pitch: ({}, {});" \
          " dist: {};" \
          " contours: {};" \
          " targets: {}".format(
        int(values['width']), int(values['height']),
        int(values['center_x']), int(values['center_y']),
        values['yaw'], values['pitch'],
        values['distance_esti'],
        int(values['contours']),
        int(values['targets']))
    with open('vision_net_values.csv', mode='a+') as vnv_file:
        vnv_writer = csv.writer(vnv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        vnv_writer.writerow([int(values['center_x']), int(values['center_y']), int(values['contours']),
                             int(values['targets'])])
        sys.stdout.write(msg)

    def value_changed(table, key, value, isNew):
        global values
        values[key] = value
        if key == 'contours':
            view()

    def start_listener():
        # During vision itit
        table.addEntryListener(value_changed, key="width")
        table.addEntryListener(value_changed, key="height")
        # Updated during match
        table.addEntryListener(value_changed, key='center_x')
        table.addEntryListener(value_changed, key='center_y')
        table.addEntryListener(value_changed, key='yaw')
        table.addEntryListener(value_changed, key='distance_esti')
        table.addEntryListener(value_changed, key='contours')
        table.addEntryListener(value_changed, key='targets')
        table.addEntryListener(value_changed, key='pitch')

    nt.NetworkTables.initialize(server='10.18.16.2')
    table = nt.NetworkTables.getTable('SmartDashboard')

    view()

    start_listener()

    while True:
        time.sleep(1)
