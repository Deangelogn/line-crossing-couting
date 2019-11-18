import numpy as np

NEG = 'N'
POS = 'P'
MID = 'M'

ACTIONS = {
    NEG + MID: "InN",
    POS + MID: "InP",
    MID + NEG: "OuN",
    MID + POS: "OuP",
    NEG + POS: "CrP",
    POS + NEG: "CrN",
}


class Zone:
    location = None
    thickness = None
    type = None
    order = None
    size_limit = None

    def __init__(self, location=100, th=5, type="horizontal",
                 ord="ascending", size_limit=640):
        self.location = location
        self.thickness = th
        self.type = type
        self.order = ord
        self.size_limit = size_limit

    def set_location(self, location):
        self.location = location

    def set_thickness(self, thickness):
        self.thickness = thickness

    def set_type(self, type):
        type_list = ["horizontal", "vertical"]
        if type in type_list:
            self.type = type
        else:
            print("Type not accepted! Try to use 'horizontal' or 'vertical' \
                   instead")

    def set_location(self, order):
        order_list = ["ascending", "descending"]
        if order in order_list:
            self.order = order
        else:
            print("Order not accepted! Try to use 'ascending' or 'descending' \
                   instead")

    def get_location(self):
        return self.location

    def get_thickness(self):
        return self.thickness

    def get_type(self):
        return self.type

    def get_order(self):
        return self.order

    def get_limits(self):
        starting_edge = self.location - self.thickness
        ending_edge = self.location + self.thickness

        if self.type == "horizontal":
            return (0, starting_edge), (self.size_limit, ending_edge)

        if self.type == "vertical":
            return (starting_edge, 0), (ending_edge, self.size_limit)

    def analyse_trajectory_in_zone(self, trajectory):
        events = []
        zone_start, zone_end = self.get_limits()
        start, end = (zone_start[1], zone_end[1]) if self.type == 'horizontal'\
                     else (zone_start[0], zone_end[0])

        trajectory = self.get_trajectory_by_type(trajectory)

        trajectory_arr = np.asarray(list(trajectory))
        zone = np.empty(trajectory_arr.shape, 'str')

        zone[trajectory_arr < start] = 'N'
        trajectory_arr[trajectory_arr < start] = -1

        zone[trajectory_arr>end] = 'P'
        trajectory_arr[trajectory_arr>end] = 1

        zone[(start <= trajectory_arr) & (trajectory_arr <= end)] = 'M'
        trajectory_arr[(start <= trajectory_arr) & (trajectory_arr <= end)] = 0

        trajectory_arr_diff = np.diff(trajectory_arr)

        for i, value in enumerate(trajectory_arr_diff):
            if value != 0:
                key = zone[i]+zone[i+1]
                events.append(ACTIONS[key])

        return events

    def get_trajectory_by_type(self, trajectory):
        trajectory_type = []

        for coords in trajectory:
            coord = coords[1] if self.type == 'horizontal' else coords[0]
            trajectory_type.append(coord)

        return trajectory_type
