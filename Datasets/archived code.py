
# def files_exist(files):
#     # NOTE: We return `False` in case `files` is empty, leading to a
#     # re-processing of files on every instantiation.
#     return len(files) != 0 and all([fs.exists(f) for f in files])

# class AirFransGeo(InMemoryDataset):
#     def __init__(self, root, dataset, indices, transform=None, pre_transform=None, pre_filter=None):
#         super().__init__(root, transform, pre_transform, pre_filter)
#         self.dataset = dataset
#         # self.raw_dir = os.path.join(root, 'raw')
#         # self.proc_dir = os.path.join(root, 'processed')
#         self.indices = indices

#         self.raw_fnames = [x + '.pkl' for x in self.dataset.extra_data['simulation_names'][:,0]]
#         self.proc_fnames = [x + '.pt' for x in self.dataset.extra_data['simulation_names'][:,0]]
#         self.raw_p = [os.path.join(root, 'raw', x) for x in self.raw_fnames]
#         self.raw_dir = os.path.join(root, 'raw')

#         # self.raw_paths = self.raw_file_names

#         self.load(self.processed_paths[0])

#     # @property
#     # def processed_dir(self):
#     #     name = 'processed'
#     #     return osp.join(self.root, name)

#     @property
#     def raw_paths(self):
#         return self.raw_p

#     @property
#     def raw_file_names(self):
#         return self.raw_fnames

#     @property
#     def processed_file_names(self):
#         return self.proc_fnames

#     def _download(self):
#         if files_exist(self.raw_paths):  # pragma: no cover
#             return

#         fs.makedirs(self.raw_dir, exist_ok=True)
#         self.download()


#     def download(self):
#         for i, sim in zip(self.indices, self.dataset.extra_data['simulation_names'][:,indices]):
#             file = open(os.path.join(self.raw_dir, sim + '.pkl'), 'wb')
#             pickle.dump(extract_dataset_by_simulation('sim', self.dataset, i), file)
#             file.close()

#     def process(self):
#         # Read data into huge `Data` list.
#         data_list = []
#         for i in self.indices:
#             file = open(os.path.join(self.raw_dir, 
#                     self.dataset.extra_data['simulation_names'][i,0]+ '.pkl'), 'wb')
#             data_list.append(pickle.load(file))
#             file.close()

#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]

#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]

#         self.save(data_list, self.processed_paths[0])

def extract_dataset_by_simulation(newdataset_name:str,
                                   dataset:AirfRANSDataSet,
                                   simulation_index:int):
    simulation_sizes = dataset.get_simulations_sizes()
    sample_sizes = [None]*len(simulation_sizes)
    start_index = 0
    for simulation_Id,simulation_size in enumerate(simulation_sizes):
        sample_sizes[simulation_Id] = range(start_index,start_index+simulation_size)
        start_index+= simulation_size
    # values=operator.itemgetter(*list(simulation_indices))(sample_sizes)
    nodes_simulation_indices = sorted([item for sublist in [sample_sizes[simulation_index]] for item in sublist])

    new_data={}
    for data_name in dataset._attr_names:
        new_data[data_name]=dataset.data[data_name][nodes_simulation_indices]
    new_extra_data={
                    'simulation_names':dataset.extra_data['simulation_names'][simulation_index],
                    'surface':dataset.extra_data['surface'][nodes_simulation_indices]
                    }
    new_dataset=type(dataset)(config = dataset.config, 
                             name = newdataset_name,
                             task = dataset._task,
                             split = dataset._split,
                             attr_names = dataset._attr_names, 
                             attr_x = dataset._attr_x , 
                             attr_y = dataset._attr_y)

    new_dataset.data=new_data
    new_dataset.extra_data=new_extra_data
    new_dataset._infer_sizes()
    return new_dataset


from airfrans.simulation import Simulation
import pyvista as pv

folder = 'airFoil2D_SST_58.831_-3.563_2.815_4.916_10.078'

sim = Simulation(DIRECTORY_NAME, folder)
print(sim.angle_of_attack)
print(sim.inlet_velocity)

# mesh_00 = pv.read(os.path.join(DIRECTORY_NAME, folder, folder+'_internal.vtu'))
mesh_00 = sim.internal
mesh_00.plot(scalars='nut', style='points')
mesh_00



from scipy.spatial.distance import cdist

dists = cdist(sim.position, surface[:,:2], metric='euclidean')
best_idx = np.argmin(dists,axis=1).T.tolist()
closest_surfaces = np.take(surface, best_idx, axis=0)

best_dist = np.linalg.norm(sim.position - closest_surfaces[:,:2], axis=1)
I = 151106
print(sim.position[I,:2])
print(closest_surfaces[I,:2])
print(sim.sdf[I])
print(best_dist[I])
print(np.argmin(dists[I]))

# sdf is closest distance to airfoil, even if not a point on the airfoil!

# Line from a to b
def delta_vector(from_v, to_v):
    return to_v-from_v

def angle_off_x_axis(a):
    # Note that both vectors begin at the origin, so we actually want them compared vs. [1,0]
    norm = np.linalg.norm(a)
    return np.arccos(a.dot(np.array([1,0])) / norm)

print(delta_vector([-.5,.5], closest_surfaces[I,:2]))
print(angle_off_x_axis(delta_vector([-.5,.5], closest_surfaces[I,:2])))