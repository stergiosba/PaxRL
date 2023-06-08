import pyglet as pg
import numpy as np
import numpy.linalg as la

class VisualEntity(pg.shapes.Circle):
    def __init__(self, id, x, y, radius, neighborhood_radius, segments=None, color=..., batch=None, group=None):
        super().__init__(x, y, radius, segments, color, batch, group)
        
        self.label = pg.text.Label(str(id),
                        font_name='Times New Roman',
                        font_size=12,
                        x=x, y=y,
                        color=(0,0,0,255),
                        anchor_x='center', anchor_y='center', batch=batch)
        self.neighborhood = pg.shapes.Circle(x, y, radius=neighborhood_radius, color=(color[0], color[1], color[2], 45), batch=batch)
        
        self.heading = pg.shapes.Line(x, y, x, y, 2, color = (255, 0, 0, 255), batch = batch)
        
    def update(self, velocity):
        #self.neighborhood.position = self.position
        self.heading.x = self.x
        self.heading.y = self.y
        
        norm_v = la.norm(velocity)
        e_v = velocity/norm_v
        self.heading.x2 = self.x+20*e_v[0]
        self.heading.y2 = self.y+20*e_v[1]
        
        self.label.position = (self.x, self.y, 0)