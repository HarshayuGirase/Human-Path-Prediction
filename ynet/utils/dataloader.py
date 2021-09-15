from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch

class SceneDataset(Dataset):
	def __init__(self, data, resize, total_len):
		""" Dataset that contains the trajectories of one scene as one element in the list. It doesn't contain the
		images to save memory.
		:params data (pd.DataFrame): Contains all trajectories
		:params resize (float): image resize factor, to also resize the trajectories to fit image scale
		:params total_len (int): total time steps, i.e. obs_len + pred_len
		"""

		self.trajectories, self.meta, self.scene_list = self.split_trajectories_by_scene(data, total_len)
		self.trajectories = self.trajectories * resize

	def __len__(self):
		return len(self.trajectories)

	def __getitem__(self, idx):
		trajectory = self.trajectories[idx]
		meta = self.meta[idx]
		scene = self.scene_list[idx]
		return trajectory, meta, scene

	def split_trajectories_by_scene(self, data, total_len):
		trajectories = []
		meta = []
		scene_list = []
		for meta_id, meta_df in tqdm(data.groupby('sceneId', as_index=False), desc='Prepare Dataset'):
			trajectories.append(meta_df[['x', 'y']].to_numpy().astype('float32').reshape(-1, total_len, 2))
			meta.append(meta_df)
			scene_list.append(meta_df.iloc()[0:1].sceneId.item())
		return np.array(trajectories), meta, scene_list


def scene_collate(batch):
	trajectories = []
	meta = []
	scene = []
	for _batch in batch:
		trajectories.append(_batch[0])
		meta.append(_batch[1])
		scene.append(_batch[2])
	return torch.Tensor(trajectories).squeeze(0), meta, scene[0]
