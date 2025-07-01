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



###################################################  PLANT  ###############################################

add_zone((-0.95, -1.4),(0.95, -1.05),"light_red")
add_zone((-0.3, -1.05),(0.3, -0.2),"light_red")
add_zone((-0.2, -0.2),(0.2, 0.55),"light_red")
add_zone((-0.3, -0.2),(-1.2, -0.55),"light_red")
add_zone((0.3, -0.2),(1.2, -0.55),"light_red")
add_zone((1.2, -0.55),(0.8, 0.1),"light_red")
add_zone((-1.2, -0.55),(-0.8, 0.1),"light_red")
add_zone((-0.2, -1.4),(0.2, -1.2),"cream")
add_zone((-0.2, 0.55),(0.2, 0.8),"light_orange")
add_zone((-1.2, 0.1),(-0.7, 0.6),"pink")
add_zone((1.2, 0.1),(0.7, 0.6),"light_purple")
add_zone((-0.95, -1.4),(-0.375, -1.05),"light_pink")
add_zone((0.95, -1.4),(0.375, -1.05),"light_brown")
add_zone((-0.9, 1.3),(0.9, 0.8),"light_yellow")
add_zone((-0.9, 1.3),(-0.45, 1.05),"light_green")
add_zone((0.9, 1.3),(0.45, 1.05),"light_blue")


add_wall((-0.95, -1.4), (-0.2, -1.35)) 
add_wall((0.2, -1.4), (0.95, -1.35))
add_wall((0.95, -1.35), (0.9, -1.05))
add_wall((-0.95, -1.35), (-0.9, -1.05))
add_wall((0.95, -1), (0.3, -1.05))
add_wall((-0.95, -1), (-0.3, -1.05))
add_wall((0.35, -1), (0.3, -0.55))
add_wall((-0.35, -1), (-0.3, -0.55))
add_wall((-0.30, -0.55), (-1.25, -0.5))
add_wall((0.30, -0.55), (1.25, -0.5))
add_wall((-1.25, -0.5), (-1.2, 0.6))
add_wall((1.25, -0.5), (1.2, 0.6))
add_wall((-1.25, 0.6), (-0.70, 0.55))
add_wall((1.25, 0.6), (0.70, 0.55))
add_wall((-0.75, 0.55), (-0.7, 0.1))
add_wall((0.75, 0.55), (0.7, 0.1))
add_wall((-0.75, 0.15), (-0.85, 0.1))
add_wall((0.75, 0.15), (0.85, 0.1))
add_wall((-0.80, 0.1), (-0.85, -0.2))
add_wall((0.8, 0.1), (0.85, -0.2))
add_wall((-0.85, -0.2), (-0.2, -0.15))
add_wall((0.85, -0.2), (0.2, -0.15))
add_wall((-0.2, -0.15), (-0.25, 0.75))
add_wall((0.2, -0.15), (0.25, 0.75))
add_wall((-0.2, 0.75), (-0.9, 0.8))
add_wall((0.2, 0.75), (0.9, 0.8))
add_wall((-0.9, 0.8), (-0.85, 1.3))
add_wall((0.9, 0.8), (0.85, 1.3))
add_wall((-0.5, 1.3), (-0.45, 1.05))
add_wall((0.5, 1.3), (0.45, 1.05))
add_wall((-0.9, 1.3), (0.9, 1.35))

################################################################################################################

with open(PATH, "w") as f:
    f.write(txt)