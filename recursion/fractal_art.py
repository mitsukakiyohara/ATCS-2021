#Fractal Art
import turtle
tr = turtle.Turtle()

#Spiral Function
def spiral(initial_length, angle, multiplier): 
    if initial_length > 200: 
        return 
    elif initial_length < 1: 
        return 
    else: 
        tr.pendown()
        tr.forward(initial_length)
        tr.left(angle)
        spiral(initial_length * multiplier, angle, multiplier)

#print(spiral(100, 90, 0.9))

#Tree Function
def tree(initial_length, height): 
    if height > 0: 
        tr.forward(initial_length)
        tr.right(30)
        #Recursive call for the right subtree 
        tree(initial_length - 15, height - 1)
        
        tr.left(60)
        #Recurisve call for the left subtree 
        tree(initial_length - 15, height - 1)

        tr.right(30)
        tr.backward(initial_length)

#Set screensize
# tr.screen.screensize(2000,2000)
# # Have turtle face upwards
# tr.right(-90)
# print(tree(200, 4))
# tr.done()

#Snowflake Function 
def snowflake(initial_length, levels): 
    if levels == 0:
        tr.forward(initial_length)
        return
    
    initial_length /= 3
    snowflake(initial_length, levels - 1)
    tr.left(60)
    snowflake(initial_length, levels - 1)
    tr.left(-120)
    snowflake(initial_length, levels - 1)
    tr.left(60)
    snowflake(initial_length, levels - 1)
    tr.left(0)

# tr.pendown()
# tr.speed("fastest")
# for i in range(3):
#     print(snowflake(280, 4))
#     tr.right(120)
# turtle.done()

    







