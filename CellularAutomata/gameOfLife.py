"""
CONWAY'S GAME OF LIFE
Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
Any live cell with two or three live neighbours lives on to the next generation.
Any live cell with more than three live neighbours dies, as if by overpopulation.
Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
"""
#---------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
#---------------------------------------------
def count_neighbours(r,c,grid,n_ROWS,n_COLS):
    counter = 0
    for offr in [r-1,r,r+1]:
        if (0 <= offr) and (offr < n_ROWS):
            for offc in [c-1,c,c+1]:
                if (0 <= offc) and (offc < n_COLS):
                    counter += grid[offr,offc]
  
    counter -= grid[r,c] #remove the cell itself from the counter because it's not a neighbour
    return counter
#---------------------------------------------
#GRID SIZE
n_ROWS = 50
n_COLS = 50
#Init plot
fig, ax = plt.subplots()

while True:
    grid = np.zeros((n_ROWS,n_COLS))
    neighbourGrid = np.zeros((n_ROWS,n_COLS))

    gridInput = input("Enter the percentage of live cells (1 to 100) in the starting grid or 0 to quit the program: ")
    if gridInput == "0":
        break
    elif gridInput.isdigit():
        gridDensity = int(gridInput)
        if not(gridDensity <= 100):
            continue #invalid input -> try again
    else:
        continue #invalid input -> try again

    #POPULATING THE GRID
    for r in range(0,n_ROWS):
        for c in range(0,n_COLS):
            chance = np.random.randint(1,100)
            if chance <= gridDensity:
                grid[r,c] = 1
            else:
                grid[r,c] = 0

    #EVOLUTION OVER TIME
    for i in range(0,100):
        #COUNTING NEIGHBOURS
        for r in range(0,n_ROWS):
            for c in range(0,n_COLS):
                neighbourGrid[r,c] = count_neighbours(r,c,grid,n_ROWS,n_COLS)

        #NEW GENERATION
        for r in range(0,n_ROWS):
            for c in range(0,n_COLS):
                n_neigh = neighbourGrid[r,c]

                if grid[r,c] == 1:  #Live cell
                    if n_neigh < 2 or 3 < n_neigh:
                        grid[r,c] = 0
                else:               #Dead cell
                    if n_neigh == 3:
                        grid[r,c] = 1

        #PLOT
        ax.cla()
        ax.imshow(grid,cmap='gray')
        ax.set_title(f"frame {i}")
        plt.pause(0.05)
