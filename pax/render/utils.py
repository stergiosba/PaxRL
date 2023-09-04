import pyglet as pg
import numpy as np
import numpy.linalg as la

neighbourhood = True
heading = True

class VisualEntity(pg.shapes.Circle):
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
            e_v = velocity/norm_v
            self.heading.x2 = self.x+20*e_v[0]
            self.heading.y2 = self.y+20*e_v[1]


class ScriptedEntity(VisualEntity):
    def __init__(self, id, x, y, radius, neighborhood_radius, segments=None, color=..., batch=None, group=None):
        super().__init__(x, y, radius, neighborhood_radius, segments, color, batch, group)

        self.label = pg.text.Label(str(id),
                font_name='Times New Roman',
                font_size=10,
                x=x, y=y,
                color=(255,255,255,255),
                anchor_x='center', anchor_y='center', batch=batch)
        
    def update(self, position, velocity):
        super().update(position, velocity)

        self.label.position = (self.x, self.y, 0)


class AgentEntity(VisualEntity):
    def __init__(self, x, y, radius, neighborhood_radius, segments=None, color=..., batch=None, group=None):
        super().__init__(x, y, radius, neighborhood_radius, segments, color, batch, group)