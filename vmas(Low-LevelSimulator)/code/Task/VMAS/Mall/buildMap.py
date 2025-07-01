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


###################################################  MALL  ###############################################

add_zone((-0.85, 0.58),(0.0,0.275),"light_red")
add_zone((0.85, 0.58),(0.0,0.275),"light_orange")
add_zone((-0.25, 0.275),(0.25,-0.35),"light_yellow")
add_zone((-0.25, -0.35),(0.25,-0.9),"light_green")
add_zone((-0.85, -0.9),(-0.22,-0.32),"light_pink")
add_zone((0.85, -0.9),(0.22,-0.32),"cream")
add_zone((-0.85, -0.32),(-0.22,0.28),"light_purple")
add_zone((0.85, -0.32),(0.22,0.28),"light_brown")
add_zone((-0.85, 1),(0,0.58),"light_blue")
add_zone((0.85, 1),(0,0.58),"pink")

add_wall((-0.85, -0.9), (-0.8, 1)) 
add_wall((-0.8, 1), (0.85, 0.95)) 
add_wall((0.85, -0.9), (0.8, 1)) 

add_wall((-0.85, -0.9), (-0.2, -0.85)) 
add_wall((0.85, -0.9), (0.2, -0.85)) 

add_wall((-0.25, -0.85), (-0.2, -0.65)) 
add_wall((0.25, -0.85), (0.2, -0.65)) 

add_wall((-0.25, -0.35), (-0.2, -0.1)) 
add_wall((0.25, -0.35), (0.2, -0.1)) 

add_wall((-0.8, -0.35), (-0.65, -0.3)) 
add_wall((0.8, -0.35), (0.65, -0.3)) 

add_wall((-0.4, -0.35), (-0.2, -0.3)) 
add_wall((0.4, -0.35), (0.2, -0.3)) 

add_wall((-0.85, 0.25), (-0.55, 0.3)) 
add_wall((0.85, 0.25), (0.55, 0.3)) 

add_wall((-0.35, 0.25), (-0.2, 0.3)) 
add_wall((0.35, 0.25), (0.2, 0.3)) 

add_wall((-0.2, 0.3), (-0.25, 0.1))
add_wall((0.2, 0.3), (0.25, 0.1))  

add_wall((-0.85, 0.55), (-0.6, 0.6)) 
add_wall((0.85, 0.55), (0.6, 0.6)) 

add_wall((-0.35, 0.55), (0.35, 0.6)) 

add_wall((-0.025, 1), (0.025, 0.825)) 
add_wall((-0.025, 0.55), (0.025, 0.675)) 

################################################################################################################



with open(PATH, "w") as f:
    f.write(txt)