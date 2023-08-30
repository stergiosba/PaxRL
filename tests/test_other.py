#%%
import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

class Drone():
    def __init__(self, index):
        self.index = index
        self.name = f"Drone{str(index)}"
        self.v_thres = 1000
        self.neighbor_radius=30
        self.origin = np.zeros(3)
        self.init_yaw = 0
        
    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"
        
    def get_neighbors(self, flock_list):
        neighbors = []
        for drone in flock_list:
            if drone.name != self.name:
                neighbors.append(drone)
        return neighbors

    def get_neighbors_dis(self, flock_list):
        neighbors = []
        for drone in flock_list:
            self_position = self.get_position()
            drone_position = self.get_neighbor_position(drone)
            distance = np.linalg.norm(self_position-drone_position)
            if ((distance>0) and (distance <= self.neighbor_radius)):
                neighbors.append(drone)
        return neighbors
    
    def get_position(self):
        pos = self.index*np.ones(3)
        return pos + self.origin
        
    def get_velocity(self):
        return 0.1*(self.index+1)*np.ones(3)
    
    def get_neighbor_position(self, other_drone):
        pos = other_drone.get_position()
        return pos + other_drone.origin
    
    def get_neighbor_velocity(self, other_drone):
        return other_drone.get_velocity()

    
class Controller(object):
    def __init__(self, drone, flock_list):#, args):
        self.drone = drone
        self.flock_list = flock_list
        self.pos2v_scale = 10
        self.sep_weight = 1
        self.ali_weight = 1
        self.coh_weight = 0.2
        self.sep_max_cutoff = 30
        self.ali_radius = 30
        self.coh_radius = 30

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"
    
    def get_sep_velocity(self):
        v_sep = 0

        neighbors = self.drone.get_neighbors(self.flock_list)
        neighbor_number = len(neighbors)
        if neighbor_number == 0:
            return 0
        count = 0
        for neighbor in neighbors:
            pos = self.drone.get_position()
            pos_neighbor = self.drone.get_neighbor_position(neighbor)
            diff = pos - pos_neighbor
            distance = np.linalg.norm(diff)
            if 0 < distance < self.sep_max_cutoff:
                normalize(diff)
                v_sep += diff/distance
                count +=1
        if count == 0:
            return 0
        else:
            v_sep /= count
        v_sep = normalize(v_sep) * self.pos2v_scale
        return v_sep

    def get_avg_neighbor_velocity_dis(self):
        neighbors_dis = self.drone.get_neighbors_dis(self.flock_list)
        neighbor_number = len(neighbors_dis)

        if neighbor_number == 0:
            return self.drone.get_velocity()
        v_neighbor = 0
        for neighbor in neighbors_dis:
            v = self.drone.get_neighbor_velocity(neighbor)
            v_neighbor += v
        v_neighbor += self.drone.get_velocity()

        return v_neighbor/(neighbor_number+1)

    def get_ali_velocity(self):
        v_neighbor_avg = self.get_avg_neighbor_velocity_dis()
        v_align = v_neighbor_avg
        # v_align = normalize(v_align) * self.v_max
        return v_align

    def get_avg_neighbor_pos_dis(self):
        neighbors_dis = self.drone.get_neighbors_dis(self.flock_list)
        neighbor_number = len(neighbors_dis)
        if neighbor_number == 0:
            return 0
        pos_neighbor = 0
        for neighbor in neighbors_dis:
            pos = self.drone.get_neighbor_position(neighbor)
            pos_neighbor += pos
        return pos_neighbor / neighbor_number

    def get_coh_velocity(self):
        pos_neighbor_avg = self.get_avg_neighbor_pos_dis()
        pos = self.drone.get_position()
        pos_coh = pos_neighbor_avg - pos
        return normalize(pos_coh) * self.pos2v_scale

    def step(self):
        v_sep = self.get_sep_velocity()
        v_ali = self.get_ali_velocity()
        v_coh = self.get_coh_velocity()
        v_total = self.sep_weight*v_sep + self.ali_weight*v_ali + self.coh_weight*v_coh
        return v_total
        


def exper(C,T):
    for _ in range(T):
        for c in C:
            c.step()

#%%
import timeit
#suspect it grows linearly with the number of parallel episodes
T=1*60 #grows linearly with the episode size

TT= []
m=1
for n in range(1,201,10):
    F = [Drone(i) for i in range(n)]
    C = [Controller(F[i],F) for i in range(n)]
    t=(timeit.timeit("exper(C,T)", setup="from __main__ import exper,C,T",number=m)/m)
    TT.append(t)
    print(f"{n}: {t}")

#%%
import matplotlib.pyplot as plt
N=np.arange(1,201,10)
model = np.poly1d(np.polyfit(N, TT, 2))
polyline = np.linspace(1, 100, 50)
plt.scatter(N,TT)
plt.plot(polyline, model(polyline))
# %%
