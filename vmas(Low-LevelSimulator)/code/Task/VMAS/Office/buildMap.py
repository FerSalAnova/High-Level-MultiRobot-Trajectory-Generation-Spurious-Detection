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


###################################################  OFFICE  ###############################################

add_zone((-1.4, 1.4), (-0.65, 0.35),"pink")
add_zone((-1.4, -1.4), (-0.65, -0.35),"light_purple")
add_zone((-0.65, 1.4), (0.4, 0.25),"light_brown")
add_zone((-0.65, -1.4), (0.4, -0.25),"cream")
add_zone((0.9, -0.3),(1.4,0.3),"light_red")
add_zone((0.9, -0.3),(-0.3,0.3),"light_orange")
add_zone((-1.05, -0.9), (-0.3, 0.85),"light_yellow")
add_zone((-1.4, -0.35), (-1.05, 0.35),"light_pink")
add_zone((0.4, -1.4), (1.4, -0.25),"light_blue")
add_zone((0.4, 1.4), (1.4, 0.25),"light_green")



add_wall((-1.4, -1.4), (-1.35, 1.35)) 
add_wall((-1.4, 1.4), (1.35, 1.35)) 
add_wall((-1.4, -1.4), (1.35, -1.35)) 
add_wall((1.4, 1.4), (1.35, 0.25)) 
add_wall((1.4, -1.4), (1.35, -0.25)) 

add_wall((-0.3, 0.3), (0.9, 0.25)) 
add_wall((-0.3, -0.3), (0.9, -0.25)) 

add_wall((-1.05, 0.9), (-0.65, 0.85)) 
add_wall((-1.05, -0.9), (-0.65, -0.85)) 

add_wall((-1.05, 0.9), (-1.0, 0.35))  
add_wall((-1.05, -0.9), (-1.0, -0.35)) 

################################################################################################################



with open(PATH, "w") as f:
    f.write(txt)