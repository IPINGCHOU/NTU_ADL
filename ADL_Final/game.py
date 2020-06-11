#%%
import os
import random 
import pygame
import numpy as np
import math

# game settings
# game window
WIN_WIDTH, WIN_HEIGHT = 500, 500
FRAME_RATE = 150
# Plane
PLANE_WIDTH, PLANE_HEIGHT = 25, 50
PLANE_VEL = 8
PLANE_HITBOX_RADIUS = 15
# Bullets
BULLET_RADIUS = 3
BULLET_VEL = 10
MAX_BULLETS = 100
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 255)

# read plane images
GAME_FOLDER = '/disk2/iping/NTU_ADL/ADL_Final'
plane_size = (PLANE_WIDTH, PLANE_HEIGHT)
PLANE_LEFT, PLANE_RIGHT, PLANE_STAND = [], [], []
PLANE_LEFT.append(pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER,'sprites','{}.png'.format(2))), plane_size))
PLANE_LEFT.append(pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER,'sprites','{}.png'.format(1))), plane_size))
PLANE_RIGHT.append(pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER,'sprites','{}.png'.format(4))), plane_size))
PLANE_RIGHT.append(pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER,'sprites','{}.png'.format(5))), plane_size))
PLANE_STAND.append(pygame.transform.scale(pygame.image.load(os.path.join(GAME_FOLDER,'sprites','{}.png'.format(3))), plane_size))


window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption('bullet hell drill')
clock = pygame.time.Clock()

class Plane(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = PLANE_WIDTH
        self.height = PLANE_HEIGHT
        self.vel = PLANE_VEL
        self.left = False
        self.right = False
        self.horizontal_move_count = 0
        self.hitbox = (self.x, self.y)
    
    def render(self, window):
        if self.left:
            if self.horizontal_move <= 5:
                window.blit(PLANE_LEFT[0], (self.x,self.y))
            else:
                window.blit(PLANE_LEFT[1], (self.x,self.y))
            self.horizontal_move += 1
        elif self.right:
            if self.horizontal_move <= 5:
                window.blit(PLANE_RIGHT[0], (self.x,self.y))
            else:
                window.blit(PLANE_RIGHT[1], (self.x,self.y))
            self.horizontal_move += 1
        else:
            window.blit(PLANE_STAND[0], (self.x,self.y))
            horizontal_move = 0
        
        # draw hitbox
        self.hitbox = (self.x, self.y)
        pygame.draw.circle(window, RED, self.hitbox, PLANE_HITBOX_RADIUS, 2)

class Bullet(object):
    def __init__(self, color, plane_x, plane_y):
        self.color = color
        self.radius = BULLET_RADIUS
        self.vel = BULLET_VEL

        if random.choice((0,1)) == 0: # bullet come from left or right
            self.x = random.choice((0,WIN_WIDTH))
            self.y = random.uniform(WIN_HEIGHT, 0)
        else: # bullet come from top or buttom
            self.x = random.uniform(WIN_WIDTH, 0)
            self.y = random.choice((0, WIN_HEIGHT))

        x_diff = plane_x - self.x
        y_diff = plane_y - self.y
        angle = math.atan2(y_diff, x_diff)
        self.change_x = math.cos(angle) * BULLET_VEL
        self.change_y = math.sin(angle) * BULLET_VEL
    
    def move(self):
        self.x += self.change_x
        self.y += self.change_y
    
    def render(self, window):
        pygame.draw.circle(window, self.color, (int(self.x), int(self.y)), self.radius)

def WindowRender():
    window.fill((0,0,0))
    plane.render(window)
    for bullet in bullets:
        bullet.render(window)

    pygame.display.update()

# init objects
plane = Plane(250,250) # init starting point of the plane
bullets = []
run = True

while run:
    clock.tick(FRAME_RATE)

    for event in pygame.event.get():  # This will loop through a list of any keyboard or mouse events.
        if event.type == pygame.QUIT: # Checks if the red button in the corner of the window is clicked
            run = False  # Ends the game loop

    while len(bullets) < MAX_BULLETS:
        bullets.append(Bullet(YELLOW, plane.x, plane.y))
    
    for bullet in bullets:
        bullet_exist = False
        if 0 <= bullet.x and bullet.x <= WIN_WIDTH :
            if 0 <= bullet.y and bullet.y <= WIN_HEIGHT:
                bullet.move()
                bullet_exist = True

        if bullet_exist == False:
            bullets.pop(bullets.index(bullet))

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and plane.x > 0: 
        plane.x -= PLANE_VEL
        plane.left = True
        plane.right = False
    elif keys[pygame.K_RIGHT] and plane.x < 500 - PLANE_VEL - PLANE_WIDTH:
        plane.x += PLANE_VEL
        plane.left = False
        plane.right = True
    else:
        plane.left, plane.right = False, False
        plane.horizontal_move = 0

    if keys[pygame.K_UP] and plane.y > 0:
        plane.y -= PLANE_VEL
    if keys[pygame.K_DOWN] and plane.y < 500 - PLANE_HEIGHT:
        plane.y += PLANE_VEL
    
    WindowRender()

pygame.quit()