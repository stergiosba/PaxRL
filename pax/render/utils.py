import pyglet as pg
import numpy as np
import numpy.linalg as la

neighbourhood = True
heading = False
label = True

class CircleEntity(pg.shapes.Circle):
    def __init__(self, x, y, radius, neighborhood_radius, segments=None, color=..., batch=None, group=None):
        super().__init__(x, y, radius, segments, color, batch, group)

        if neighbourhood:
            self.neighborhood = pg.shapes.Circle(x, y, radius=neighborhood_radius, color=(color[0], color[1], color[2], 45), batch=batch)
        
        if heading:
            self.heading = pg.shapes.Line(x, y, x, y, 2, color = (255, 0, 0, 255), batch = batch)
    
    def update(self, position, velocity):
        self._x, self._y = position
        self._update_translation()
        if neighbourhood:
            self.neighborhood.position = position

        if heading:
            self.heading.x = self.x
            self.heading.y = self.y
            
            norm_v = la.norm(velocity)
            if norm_v == 0:
                e_v = np.array([0, 0])
            else:
                e_v = velocity/norm_v
            self.heading.x2 = self.x+norm_v*e_v[0]
            self.heading.y2 = self.y+norm_v*e_v[1]

class TriangleEntity(pg.shapes.Triangle):
    def __init__(self, x, y, radius, neighborhood_radius, color=..., batch=None, group=None):
        super().__init__(x, y-radius/2, x, y+radius/2,x+radius*np.sqrt(3)/2, y, color, batch, group)
        self.radius = radius
        if neighbourhood:
            self.neighborhood = pg.shapes.Circle(x, y, radius=neighborhood_radius, color=(color[0], color[1], color[2], 45), batch=batch)
        
        if heading:
            # When the agents are spawned into the environment the heading arrow is not visible on purpose.
            self.heading = pg.shapes.Line(x, y, x, y, 2, color = (255, 0, 0, 255), batch = batch)
    
    def update(self, position, velocity):
        self.x = position[0]
        self.y = position[1]-self.radius/2
        self.x2 = position[0]
        self.y2 = position[1] + self.radius/2
        self.x3 = position[0] + self.radius*np.sqrt(3)/2
        self.y3 = position[1]

        if neighbourhood:
            self.neighborhood.position = position

        if heading:
            norm_v = la.norm(velocity)
            if norm_v == 0:
                e_v = np.array([0, 0])
            else:
                e_v = velocity/norm_v

            self.heading.x = self.x2
            self.heading.y = self.y2
            self.heading.x2 = self.x2+norm_v*e_v[0]
            self.heading.y2 = self.y2+norm_v*e_v[1]


class ScriptedEntity(CircleEntity):
    def __init__(self, id, x, y, radius, neighborhood_radius, segments=None, color=..., batch=None, group=None):
        super().__init__(x, y, radius, neighborhood_radius, segments, color, batch, group)

        if label:
            self.label = pg.text.Label(str(id),
                    font_name='Times New Roman',
                    font_size=10,
                    x=x, y=y,
                    color=(255,255,255,255),
                    anchor_x='center', anchor_y='center', batch=batch)
        
    def update(self, position, velocity):
        super().update(position, velocity)
        if label:
            self.label.position = (self.x, self.y, 0)


class AgentEntity(CircleEntity):
    def __init__(self, x, y, radius, neighborhood_radius, segments=None, color=..., batch=None, group=None):
        super().__init__(x, y, radius, neighborhood_radius, segments, color, batch, group)