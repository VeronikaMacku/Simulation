"""
Simulation: Pool (Eight-ball): First shot
Subject: FSI - Simulace dynamických systémů
Student: Veronika Macků, 200037
"""
#--------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
#--------------------------------------------------------------------
#SIMULATION DATA
t_start = 0
t_stop = 3.5
dt = 0.01
t = np.arange(t_start,t_stop,dt)

#BALL
m = 0.17             #weight [kg]
r = 28.575           #radius [mm]

#POOL TABLE
WIDTH = 991.0               #[mm]   #Playing area
LENGTH = 1981.0             #[mm]
f_bc = 0.2                  #ball-cloth sliding friction coefficient
g = 9.81                    #[m/s^2]
FT = f_bc*m*g               #friction force[N] in motion
aT = np.floor(FT/m*1000)    #friction (de)acceleration [mm/s^2]

#CUEBALL'S INITIAL POS
pos0x = (LENGTH/8) * 2     #[mm]
pos0y = (WIDTH/4) * 2      #[mm]

#FIRSTBALL'S INITIAL POS
posFirst0x = (LENGTH/8) * 6
posFirst0y = (WIDTH/4) * 2
#--------------------------------------------------------------------
#CREATE BALLS
class Ball():
    def __init__(self):
        self.xy = np.array([0,0],dtype=float)
        self.v = np.array([0,0],dtype=float)
        self.a = np.array([0,0],dtype=float)

    def update(self):
        if np.abs(self.v[0]) < (aT*dt):
            self.a[0] = 0.0
            self.v[0] = 0.0
        else:
            #must deaccelerate in opposite direction to v
            self.a[0] = np.sign(self.v[0])*(-aT)*np.cos(np.arctan(np.abs(self.v[1]/self.v[0])))

        if np.abs(self.v[1]) < (aT*dt):
            self.a[1] = 0.0
            self.v[1] = 0.0
        else:
            if np.abs(self.v[0]) < 0.1:   #To ensure no divide by zero will happen
                self.a[1] = np.sign(self.v[1])*(-aT)*np.sin(np.pi/2)
            else:
                self.a[1] = np.sign(self.v[1])*(-aT)*np.sin(np.arctan(np.abs(self.v[1]/self.v[0])))

        self.v += self.a*dt
        self.xy += self.v*dt

        self.ifInCollisionWithTable()

    def ifInCollisionWithTable(self):
        if ((self.xy[0] - r) < 0):  #LEFT
            self.v[0] *= -1
            self.xy[0] = r
        if ((self.xy[0] + r) >LENGTH):  #RIGHT
            self.v[0] *= -1
            self.xy[0] = LENGTH - r
        if ((self.xy[1] - r) < 0):  #BOTTOM
            self.v[1] *= -1
            self.xy[1] = r
        if ((self.xy[1] + r) >WIDTH):   #TOP
            self.v[1] *= -1
            self.xy[1] = WIDTH - r

ballsList = []
for i in range(0,16):
    ballsList.append(Ball())

#BALLS POS during simulation
ballsx=np.zeros((16,len(t)),dtype=float)
ballsy=np.zeros((16,len(t)),dtype=float)

#CUE BALL velocity and acceleration
cueballvxy = np.zeros((2,len(t)),dtype=float)
cueballaxy = np.zeros((2,len(t)),dtype=float)
#--------------------------------------------------------------------
#RESET POS,VEL,ACC
def resetBalls():
    #POSITION
    ballsList[0].xy = np.array([pos0x,pos0y])    #CueBall

    offx = np.sqrt(3)*r     #offx = 2*r*cos(30°)
    offy = r

    for i in range(1,6):
        ballsList[i].xy = np.array([posFirst0x + (i-1)*offx,posFirst0y + (i-1)*offy])

    for i in range(6,10):
        ballsList[i].xy = np.array([posFirst0x + (i-6)*offx + offx,posFirst0y + (i-6)*offy - offy])

    for i in range(10,13):
        ballsList[i].xy = np.array([posFirst0x + (i-10)*offx + 2*offx,posFirst0y + (i-10)*offy - 2*offy])

    for i in range(13,15):
        ballsList[i].xy = np.array([posFirst0x + (i-13)*offx + 3*offx,posFirst0y + (i-13)*offy - 3*offy])

    ballsList[15].xy = np.array([posFirst0x + 4*offx,posFirst0y - 4*offy])

    #VELOCITY
    for i in range(0,len(ballsList)):
        ballsList[i].v = np.array([0.0,0.0])
        ballsList[i].a = np.array([0.0,0.0])
#--------------------------------------------------------------------
#GETTING START VALUES
def getCueBallStartVel(v0,angle):
    vx0 = v0*np.cos(angle)
    vy0 = v0*np.sin(angle)

    return [vx0, vy0]

#OPTIMALIZATION FOR SEARCHING FOR POSSIBLE COLLISIONS - Only balls in the same grid sector are considered in possible collisions instead of all possible ball pairs
def isOverlapped(in1,in2):
    return max(0,min(in1[1],in2[1]) - max(in1[0],in2[0]))

def uniformGridPosColls(balls_sorted,LENGTH,WIDTH):
    grid = [[[],[],[],[],[],[],[],[]],
            [[],[],[],[],[],[],[],[]],
            [[],[],[],[],[],[],[],[]],
            [[],[],[],[],[],[],[],[]],]

    rows = len(grid)
    cols = len(grid[0])

    offx = LENGTH/cols
    offy = WIDTH/rows

    #FINDING WHICH BALLS ARE IN WHICH SECTIONS
    for i in range(0,len(balls_sorted)):
        cx = balls_sorted[i][1]
        cy = balls_sorted[i][2]
        x_span = (cx-r,cx+r)
        y_span = (cy-r,cy+r)

        for k in range(0,rows):
            if isOverlapped(y_span,(k*offy,(k+1)*offy)):
                for l in range(0,cols):
                    if isOverlapped(x_span,(l*offx,(l+1)*offx)):
                        grid[k][l].append(i)

    #FINDING ALL POSSIBLE COLLISIONs MULTIPLES
    posCollisions = []
    for k in range(0,rows):
        for l in range(0,cols):
            if len(grid[k][l]) > 1:
                if grid[k][l] not in posCollisions:
                    posCollisions.append(grid[k][l])

    #FINDING ALL POSSIBLE COLLISIONs PAIRS
    posCollisionPairs = []
    for multiple in posCollisions:
        for i in range(0,len(multiple)):
            for j in range(i+1,len(multiple)):
                pair = [multiple[i],multiple[j]]
                if pair not in posCollisionPairs:
                    posCollisionPairs.append(pair)

    return posCollisionPairs

#CHECKING WHETHER BALLS ARE TRULY IN COLLISION
def isInCollision(ball1,ball2):
    #Pythagoras Theorem
    ab = np.abs(ball1.xy - ball2.xy)
    c = np.sqrt(ab[0]**2+ab[1]**2)

    #If collision == true, the distance of ball centres must be <= the sum of their radiuses
    return c <= (r + r)

#RECALCULATING VELOCITY AND POSITION AFTER COLLISION
def calcCollision(b1,b2):
    xy1 = b1.xy
    v1 = b1.v
    xy2 = b2.xy
    v2 = b2.v
    m1,m2 = m,m

    norm1 = np.sum((xy1 - xy2)**2)
    norm2 = np.sum((xy2 - xy1)**2)

    v1final = v1- 2*m2/(m1+m2)*(np.dot(v1-v2,xy1-xy2))/norm1*(xy1-xy2)
    v2final = v2- 2*m1/(m1+m2)*(np.dot(v2-v1,xy2-xy1))/norm2*(xy2-xy1)

    b1.v = v1final
    b2.v = v2final

    #FIX BALL POSITION - otherwise the balls could still be colliding in the next time step
    difxy = xy1 - xy2
    dist = np.linalg.norm(difxy)                      
    offsetxy = (((r + r) - dist)/2)*(difxy/dist)      #(difxy/dist) represents a normed vector (sum of squared elements == 1)

    b1.xy += offsetxy
    b2.xy += -offsetxy

#SIMULATE
def simulate(v0,v0_angle):
    #ballsList[0].v = np.array(hitCueBall(F,np.deg2rad(Fangle)))
    ballsList[0].v = np.array(getCueBallStartVel(v0,np.deg2rad(v0_angle)))

    #SAVE ALL POSITIONS
    for i in range(0,16):
        ballsx[i,0] = ballsList[i].xy[0]
        ballsy[i,0] = ballsList[i].xy[1]

    #SAVE CUEBALL VELOCITY + ACCELERATION
    cueballvxy[:,0] = ballsList[0].v
    cueballaxy[:,0] = ballsList[0].a

    #CALCULATIONS
    for ti in range(0,len(t)):
        #SAVE ALL POSITIONS
        for i in range(0,16):
            ballsx[i,ti] = ballsList[i].xy[0]
            ballsy[i,ti] = ballsList[i].xy[1]

        #SAVE CUEBALL VELOCITY + ACCELERATION
        cueballvxy[:,ti] = ballsList[0].v
        cueballaxy[:,ti] = ballsList[0].a

        #CREATE BALL LIST with their (index,posx,posy) for finding potential collisions
        ballsPos = []
        for i in range(0,16):
            ballsPos.append((i,ballsx[i,ti],ballsy[i,ti]))

        #FIND ALL COLLISIONS
        possibleCollPairs = uniformGridPosColls(ballsPos,LENGTH,WIDTH)    #Optimizing for collision finding
        for pair in possibleCollPairs:
            b1 = pair[0]
            b2 = pair[1]
            if isInCollision(ballsList[b1],ballsList[b2]):
                if np.sqrt(ballsList[b1].v[0]**2 + ballsList[b1].v[1]**2) > 0.0 or np.sqrt(ballsList[b2].v[0]**2 + ballsList[b2].v[1]**2) > 0.0:
                    calcCollision(ballsList[b1],ballsList[b2])

        for i in range(0,16):
            ballsList[i].update()
#--------------------------------------------------------------------
resetBalls()
simulate(0,0.0)
#--------------------------------------------------------------------
#PLOT
figheight = 5
figlength = 10 + 10/6*4

fig = plt.figure(figsize=(figlength,figheight),tight_layout=True)
gs = gridspec.GridSpec(2, 3, figure = fig, width_ratios=[0.1,0.6,0.3])

#MAIN figure
ax0 = fig.add_subplot(gs[:, 1],facecolor="yellow")
ax0.set_title("Pool - Eight-ball - First shot simulation")
plt.xlim((0,LENGTH))
plt.ylim((0,WIDTH))
ax0.set_xlabel("pos x [mm]")
ax0.set_ylabel("pos y [mm]")

#Create Background = Pool TABLE
for i in range (0,3):
    c_x = i*(LENGTH/2)
    for j in range (0,2):
        c_y = j*WIDTH
        pocket = mp.Circle((c_x,c_y),r*1.75,  fill = True, color = "black")
        plt.gca().add_patch(pocket)

#1 Acceleration
ax1 = fig.add_subplot(gs[0, 2])
ax1.set_title("Cue Ball Velocity and Acceleration")
ax1.plot(t,cueballaxy[0,:],'b',t,cueballaxy[1,:],'r')
ax1.legend(["ax","ay"],loc=1)
ax1.set_ylabel("Acceleration ax,ay [mm/s^2]")
ax1.set_xlabel("Time [s]")
ax1.set_xlim(t_start,t_stop)

#2 Velocity
ax2 = fig.add_subplot(gs[1, 2])
ax2.plot(t,cueballvxy[0,:],'b',t,cueballvxy[1,:],'r')
ax2.legend(["vx","vy"],loc=1)
ax2.set_ylabel("Velocity vx,vy [mm/s]")
ax2.set_xlabel("Time [s]")
ax2.set_xlim(t_start,t_stop)

fig.align_labels()

#ANIMATION
line_a_t, = ax1.plot([], [], 'g')
line_v_t, = ax2.plot([], [], 'g')

ms = 16
line, = ax0.plot([], [], 'wo', markersize = ms)
line2, = ax0.plot([], [], 'ko', markersize = ms)
line3, = ax0.plot([], [], 'ro', markersize = ms)
line4, = ax0.plot([], [], 'bo', markersize = ms)
time_template = 'time = %.1fs'
time_text = ax0.text(0.05, 0.9, '', transform=ax0.transAxes)

cx = pos0x
cy = pos0y
arrow = ax0.text(cx, cy, "          ", ha="center", va="center", rotation=0.0, size=10, bbox=dict(boxstyle="rarrow,pad=0.3", fc="r", ec="k"))
arrow.set_visible(False)

def init_animate():
    ax1.clear()
    ax1.set_title("Cue Ball Velocity and Acceleration")
    ax1.plot(t,cueballaxy[0,:],'b',t,cueballaxy[1,:],'r')
    ax1.legend(["ax","ay"],loc=1)
    ax1.set_ylabel("Acceleration ax,ay [mm/s^2]")
    ax1.set_xlabel("Time [s]")
    ax1.set_xlim(t_start,t_stop)
    ax1.figure.canvas.draw()

    ax2.clear()
    ax2.plot(t,cueballvxy[0,:],'b',t,cueballvxy[1,:],'r')
    ax2.legend(["vx","vy"],loc=1)
    ax2.set_ylabel("Velocity vx,vy [mm/s]")
    ax2.set_xlabel("Time [s]")
    ax2.set_xlim(t_start,t_stop)
    ax2.figure.canvas.draw()

    return []

def animate(j):
    #BALL DRAWING
    thisx_cue = [ballsx[0,j]]
    thisy_cue = [ballsy[0,j]]

    thisx_eight = [ballsx[7,j]]
    thisy_eight = [ballsy[7,j]]

    thisx_red = [ballsx[1,j],ballsx[2,j],ballsx[4,j],ballsx[10,j],ballsx[11,j],ballsx[12,j],ballsx[15,j]]
    thisy_red = [ballsy[1,j],ballsy[2,j],ballsy[4,j],ballsy[10,j],ballsy[11,j],ballsy[12,j],ballsy[15,j]]

    thisx_blue = [ballsx[3,j],ballsx[5,j],ballsx[6,j],ballsx[8,j],ballsx[9,j],ballsx[13,j],ballsx[14,j]]
    thisy_blue = [ballsy[3,j],ballsy[5,j],ballsy[6,j],ballsy[8,j],ballsy[9,j],ballsy[13,j],ballsy[14,j]]

    line.set_data(thisx_cue, thisy_cue)
    line2.set_data(thisx_eight, thisy_eight)
    line3.set_data(thisx_red, thisy_red)
    line4.set_data(thisx_blue, thisy_blue)
    time_text.set_text(time_template % (j*dt))

    return line, line2, line3, line4, time_text

def animate_round(i, *fargs):
    if fargs:    #if fargs == non_empty_tuple -> isIdleRound -> will draw arrow
        j = len(t) - 1
        angle = menuSim.angle

        #Arrow drawing
        arrow.set_x(ballsx[0,j] - 120*np.cos(np.deg2rad(angle)))
        arrow.set_y(ballsy[0,j] - 120*np.sin(np.deg2rad(angle)))
        arrow.set_rotation(angle)
        arrow.set_visible(True)

        line, line2, line3, line4, time_text = animate(j)
        return line, line2, line3, line4, time_text, arrow

    line, line2, line3, line4, time_text = animate(i)
    return line, line2, line3, line4, time_text

class poolAnimation(animation.FuncAnimation):
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, *, cache_frame_data=True, **kwargs):

        super().__init__(fig, func, frames, init_func, fargs, save_count, cache_frame_data = cache_frame_data, **kwargs)

    def pause(self):
        """Pause the animation."""
        self.event_source.stop()
        if self._blit:
            for artist in self._drawn_artists:
                artist.set_animated(False)

    def resume(self):
        """Resume the animation."""
        self.event_source.start()
        if self._blit:
            for artist in self._drawn_artists:
                artist.set_animated(True)

    def restart(self):
        """Restart animation from first frame."""
        self._args = () # _args (== fargs in func) = () -> not an idleRound -> sim animation runs, arrow isn't drawn
        self._init_draw()
        self.frame_seq = self.new_frame_seq()

    def _step(self, *args):
        """Gets the next frame in sequence to be drawn."""
        still_going = animation.Animation._step(self, *args)
        if not still_going and self.repeat:
            self._args = (1,1)     # _args (== fargs in func) -> non_empty_tuple -> next round is idle round -> arrow will be drawn
            self._init_draw()
            self.frame_seq = self.new_frame_seq()
            self.event_source.interval = self._repeat_delay
            return True
        else:
            self.event_source.interval = self._interval
            return still_going
 
ani = poolAnimation(fig, animate_round,frames=len(t), fargs = (1,1), init_func=init_animate,interval=dt*100, blit=True, repeat = True)

#MENU
start_vel = 5000.0
class Menu:
    def __init__(self,v0,ani):
        self.v0 = v0
        self.angle = 0.0
        self.ani = ani

    def Simulate(self, event):
        self.ani.pause()
        simulate(self.v0,self.angle)
        self.ani.restart()
        self.ani.resume()

    def SimulateReset(self, event):
        self.ani.pause()
        resetBalls()
        simulate(0.0,0.0)
        self.ani.restart()
        self.ani.resume()

    def SimulateReplay(self, event):
        self.ani.pause()
        self.ani.restart()
        self.ani.resume()

    def Vel0Change(self, val):
        self.v0 = val

    def AngleChange(self, val):
        self.angle = val

gs01 = gs[0:,0].subgridspec(2, 1)
gs01Button = gs01[0,0].subgridspec(3, 1)
gs01Slider = gs01[1,0].subgridspec(3, 1, height_ratios=[0.45,0.1,0.45])
menuSim = Menu(start_vel,ani)

axMenuSimulate = fig.add_subplot(gs01Button[0, 0])
bSimulate = Button(axMenuSimulate, 'Simulate')
bSimulate.on_clicked(menuSim.Simulate)

axMenuSimulateReset = fig.add_subplot(gs01Button[1, 0])
bSimulateReset = Button(axMenuSimulateReset, 'Reset')
bSimulateReset.on_clicked(menuSim.SimulateReset)

axMenuSimulateReplay = fig.add_subplot(gs01Button[2, 0])
bSimulateReplay = Button(axMenuSimulateReplay, 'Replay')
bSimulateReplay.on_clicked(menuSim.SimulateReplay)

axMenuVel0 = fig.add_subplot(gs01Slider[0, 0])
sVel0 = Slider(axMenuVel0, 'Starting velocity v0 [mm/s]',0.0,10000.0,start_vel,orientation = "vertical")
sVel0.on_changed(menuSim.Vel0Change)

axMenuAngle = fig.add_subplot(gs01Slider[2, 0])
sAngle = Slider(axMenuAngle, 'Angle phi0 [°]',-180.0,180.0,0,orientation = "vertical")
sAngle.on_changed(menuSim.AngleChange)

plt.show()