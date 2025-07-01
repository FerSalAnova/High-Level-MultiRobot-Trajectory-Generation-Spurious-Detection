import os
PATH = os.path.dirname(os.path.abspath(__file__)) + "/map.txt"
txt = ""
def add_wall(corner_1, corner_2):
    global txt
    txt += "wall,{:.2f},{:.2f},{:.2f},{:.2f}\n".format(
        corner_1[0], corner_1[1], corner_2[0], corner_2[1]
    )

def add_room(corner_1, corner_2, open=""):
    x_min = min(corner_1[0], corner_2[0])
    x_max = max(corner_1[0], corner_2[0])
    y_min = min(corner_1[1], corner_2[1])
    y_max = max(corner_1[1], corner_2[1])

    for side in ["top", "down", "left", "right"]:
        if open == side:
            continue
        if side=="top":
            add_wall((x_min, y_max), (x_max, y_max-0.05))
        elif side=="down":
            add_wall((x_min, y_min), (x_max, y_min+0.05))
        elif side=="left":
            add_wall((x_min, y_min), (x_min+0.05, y_max))
        elif side=="right":
            add_wall((x_max, y_min), (x_max-0.05, y_max))

def add_zone(corner_1, corner_2, color):
    global txt
    txt += "zone,{:.2f},{:.2f},{:.2f},{:.2f},{}\n".format(
        corner_1[0], corner_1[1], corner_2[0], corner_2[1], color
    )


###################################################  HOUSE  ###############################################


# add_zone((-1.4, -1.4), (1.4, -1.0),"light_blue")
# add_zone((-0.8, -1.4), (-1.4, 0.3),"light_blue")
# add_zone((-1.4, -1.4), (-1.05, -1.1),"light_red")
# add_zone((-1.4, 1.4), (0.65, 0.3),"pale_blue")
# add_zone((1.4, 1.4), (0.65, 0.875),"pale_green")
# add_zone((-0.65,-1.0), (0.3, -0.55),"pink")
# add_zone((0.3,-1.0), (0.75, -0.55),"purple")
# add_zone((0.75,-1.0), (1.35, -0.55),"blue")
# add_zone((-0.8,-0.45), (-0.3, 0.0),"light_purple")
# add_zone((-0.25,-0.2), (0.4, 0.3),"light_orange")
# add_zone((0.6,0.35), (1.4, 0.875),"light_yellow")


# add_wall((-1.4, -1.4), (-1.35, 1.35)) 
# add_wall((-1.4, 1.4), (1.35, 1.35)) 
# add_wall((-1.05, -1.4), (1.4, -1.35)) 
# add_wall((1.4, 1.4), (1.35, -1.35)) 

# add_wall((-0.8, -1.0), (-0.75, -0.35)) 
# add_wall((-0.8, -0.1), (-0.75, 0.3)) 

# add_wall((-0.8, -1.0), (-0.4, -0.95)) 
# add_wall((-0.2, -1.0), (0.4, -0.95)) 
# add_wall((0.6, -1.0), (0.9, -0.95))
# add_wall((1.1, -1.0), (1.35, -0.95))

# add_room((-0.65,-1.0), (0.3, -0.55), open="down")
# add_room((0.3,-1.0), (0.75, -0.55), open="down")
# add_room((0.75,-1.0), (1.35, -0.55), open="down")

# add_room((-0.8,-0.45), (-0.3, 0.0), open="left")

# add_wall((-0.75, 0.3), (-0.1, 0.25)) 
# add_wall((0.2, 0.3), (0.6, 0.25)) 

# add_wall((0.6, 0.25), (0.65, 0.55)) 
# add_wall((0.6, 0.75), (0.65, 0.95)) 
# add_wall((0.6, 1.15), (0.65, 1.4)) 

# add_room((-0.25,-0.2), (0.4, 0.3), open="top")

# add_room((0.6,0.35), (1.4, 0.875), open="left")

################################################################################################################

add_zone((-1.4, -1.4), (-0.8, -1.05),"light_red")
add_zone((-1.4, -1.05), (1.4, -0.675),"light_orange")
add_zone((-1.4, 1.4), (0.925, 0.475),"light_pink")
add_zone((-1.4, -1.05), (-0.775, 0.475),"light_orange")
add_zone((1.4, -0.675), (0.75,-0.3),"light_yellow")
add_zone((0.75, -0.675), (0.15,-0.3),"light_green")
add_zone((0.15, -0.675), (-0.8,-0.3),"light_purple")
add_zone((-0.775, 0.475), (0.0,-0.3),"light_blue")
add_zone((0.0,0.0), (0.9,0.5),"pink")
add_zone((1.4,1.4), (0.925,0.95),"light_brown")
add_zone((0.925,0.95),(1.4,0.5),"cream")

add_wall((-1.4, -1.4), (-1.35, 1.35))
add_wall((-1.4, 1.35), (1.35, 1.4))
add_wall((-0.8, -1.4), (-0.75, -1.1))
add_wall((-0.8, -1.1), (1.4, -1.05))
add_wall((1.4, -1.05), (1.35, -0.3))
add_wall((1.4, -0.3), (-0.75, -0.35))
add_wall((0.75, -0.35), (0.7, -0.65))
add_wall((0.895, -0.7), (0.6, -0.65))
add_wall((0.15, -0.35), (0.1, -0.65))
add_wall((0.3, -0.7), (-0.1, -0.65))
add_wall((-0.8, -0.7), (-0.5, -0.65))
add_wall((-0.8, -0.7), (-0.75, -0.1))
add_wall((0.0, -0.3), (0.05, 0.45))
add_wall((0.3, 0.45), (-0.75, 0.5))
add_wall((-0.8, 0.2), (-0.75, 0.5))
add_wall((0.55, 0.45), (1.4, 0.5))
add_wall((1.4, 0.5), (1.35, 1.4))
add_wall((0.0, 0.0), (0.9, 0.05))
add_wall((0.9, 0.05), (0.85, 0.5))
add_wall((1.35, 0.9), (0.95, 0.95))
add_wall((0.95, 0.75), (0.90, 1.1))

################################################################################################################

with open(PATH, "w") as f:
    f.write(txt)