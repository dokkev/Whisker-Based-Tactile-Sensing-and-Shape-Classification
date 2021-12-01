# coding: utf-8
# by Joao Bueno
from random import randrange, choice
import pygame
import numpy as np


SIZE = (1260, 720)
AXIS = (800, 620)
BGCOLOR = (255,255,255)

# Define the colors we will use in RGB format
BLACK = pygame.Color(  0,   0,   0, 0)
WHITE = pygame.Color(255, 255, 255, 0)
BLUE =  pygame.Color(  0,   0, 255, 0)
GREEN = pygame.Color(  0, 255,   0, 0)
RED =   pygame.Color(255,   0,   0, 0)
YELLOW = pygame.Color(255,   255,   0, 0)
offset = 50
xoffset = 150

class Plot:
    def __init__(self,size,pos,screen,title=""):
        self.screen = screen
        self.surface = pygame.Surface(size)
        self.axis = size
        self.x = pos[0]
        self.y = pos[1]
        self.xmargin = [size[0]*0.15,size[0]*0.01]
        self.ymargin = size[1]*0.2
        self.yspace = size[1]*0.2
        self.xspace = size[0]*0.07

        self.xmin = 0
        self.xmax = 10
        self.ymin = 0
        self.ymax = 50
        self.yzero = (self.axis[1]-2*self.ymargin)/2

        self.xscale = (self.axis[0]-np.sum(self.xmargin))/np.abs(self.xmax-self.xmin)
        self.yscale = (self.axis[1]-2*self.ymargin)/np.abs(self.ymax-self.ymin)

        self.previous = [self.xmargin[0],self.axis[1]-self.ymargin-self.yzero]

        self.xlabel = ""
        self.ylabel = ""
        self.title = title

        self.surface.fill(BGCOLOR)

    def draw_axis(self):
        self.surface.fill(BGCOLOR)
        pygame.draw.aaline(self.surface,BLACK,[self.xmargin[0],self.axis[1]-self.ymargin],[self.axis[0]-self.xmargin[1],self.axis[1]-self.ymargin],2)
        pygame.draw.aaline(self.surface,BLACK,[self.xmargin[0],self.ymargin],[self.xmargin[0],self.axis[1]-self.ymargin],2)
        pygame.draw.aaline(self.surface,(0,0,0,0.5),[self.xmargin[0],self.axis[1]-self.ymargin-self.yzero],[self.axis[0]-self.xmargin[1],self.axis[1]-self.ymargin-self.yzero],2)

        y = 0
        while np.abs(y)<(self.axis[1]/2-self.ymargin):
            self.draw_ytick(self.yzero+y)
            self.draw_ytick(self.yzero-y)
            y += self.yspace

        x = 0.
        while x<(self.axis[0]-2*self.xmargin[0]):
            self.draw_xtick(x)
            x += self.xspace

    def draw_xtick(self,x):
        pygame.draw.aaline(self.surface,BLACK,[self.xmargin[0]+x,self.axis[1]-self.ymargin+5],[self.xmargin[0]+x,self.axis[1]-self.ymargin-5],2)
       

    def draw_ytick(self,y):
        pygame.draw.aaline(self.surface,BLACK,[self.xmargin[0]-5,self.axis[1]-self.ymargin-y],[self.xmargin[0]+5,self.axis[1]-self.ymargin-y],2)

    def xticklabel(self,x):
        
        xvalue = x/self.xscale + self.xmin
        string = '{0:.1f}'.format(xvalue)

        myfont = pygame.font.SysFont("arial", 12)
        label = myfont.render(string, 1, (0,0,0))
        self.screen.blit(label, (self.x+self.xmargin[0]+x-10,self.y+self.axis[1]-self.ymargin+15))

    def yticklabel(self,y):
        
        yvalue = (y-self.yzero)/self.yscale
        string = '{0:.1e}'.format(yvalue)

        myfont = pygame.font.SysFont("arial", 12)
        label = myfont.render(string, 1, (0,0,0))
        self.screen.blit(label, (self.x+self.xmargin[0]-55,self.y+self.axis[1]-self.ymargin-y-5))

    def draw_ticklabels(self):

        x = 0.
        while x<(self.axis[0]-np.sum(self.xmargin)):
            self.xticklabel(x)
            x += self.xspace

        y = 0
        while np.abs(y)<(self.axis[1]/2-self.ymargin):
            self.yticklabel(self.yzero+y)
            self.yticklabel(self.yzero-y)
            y += self.yspace

    def clear(self):
        self.draw_axis()

    def plot(self,x, y,color=BLACK):
           
            
        # Redraw the background
        if x > self.xmax:
            span = self.xmax-self.xmin
            self.xmin = self.xmax
            self.xmax = self.xmax+span
            self.draw_axis()
            self.previous[0] = self.xmargin[0]

        #Scale data
        xp = (x-self.xmin)*self.xscale + self.xmargin[0]
        yp = self.axis[1] - self.ymargin - self.yzero-(y)*self.yscale
        if np.isnan(xp):
            xp = np.nan_to_num(xp)
        if np.isnan(yp):
            yp = np.nan_to_num(yp)
 
        
        current = [int(xp), int(yp)]
        # self.surface.set_at((int(xp), int(yp)), BLACK)
        # pygame.draw.circle(self.surface,BLACK,current,3)
        
        pygame.draw.aaline(self.surface,color,self.previous,current,3)

        self.previous = current

    def draw_labels(self):

        myfont = pygame.font.SysFont("arial", 15)
        label = myfont.render(self.xlabel, 1, (0,0,0))
        self.screen.blit(label, (self.x+self.axis[0]/2,self.y+self.axis[1]+20))

        myfont = pygame.font.SysFont("arial", 20)
        label = myfont.render(self.ylabel, 1, (0,0,0))
        self.screen.blit(label, (self.x+self.xmargin[0]/8,self.y+self.axis[1]/2-20))

        myfont = pygame.font.SysFont("arial", 20)
        label = myfont.render(self.title, 1, (0,0,0))
        self.screen.blit(label, (self.x+self.axis[0]/2,self.y+self.ymargin-20))

    

class Graph:
    def __init__(self,screen):
        # Make a screen to see
        
        self.screen = screen

        w, h = pygame.display.get_surface().get_size()
        self.offset = [0.01*w,0.05*h]
        self.winsize = (w-2*self.offset[0],h-2*self.offset[1])
        self.subplots = [Plot(self.winsize,(0,0),screen)]

        
        
    def add_subplots(self,rows,cols):
        self.subplots = []
        i = 0
        h = self.winsize[1]/rows
        w = self.winsize[0]/cols
        for r in range(rows):
            for c in range(cols):
                x = c*w
                y = r*h
                p = Plot((w,h),(self.offset[0]+x,self.offset[1]+y),self.screen)
                self.subplots.append(p)
                

    def xlabel(self,num,label):
        self.subplots[num].xlabel = label

    def ylabel(self,num,label):
        self.subplots[num].ylabel = label

    def set_title(self,num,label):
        self.subplots[num].title = label

    def plot(self,num,x,y,color=BLACK):
        # Put the surface we draw on, onto the screen
        self.subplots[num].plot(x,y,color)
        self.screen.blit(self.subplots[num].surface,(self.subplots[num].x,self.subplots[num].y))
        self.subplots[num].draw_ticklabels()
        self.subplots[num].draw_labels()

    def update(self):        
        pygame.display.flip()

    def flush(self):
        for p in self.subplots:
            p.clear()

