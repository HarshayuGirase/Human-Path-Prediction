import numpy as np
import pandas as pd
import os
import cv2
from copy import deepcopy


def load_SDD(path='data/SDD/', mode='train'):
	'''
	Loads data from Stanford Drone Dataset. Makes the following preprocessing:
	-filter out unnecessary columns (e.g. generated, label, occluded)
	-filter out non-pedestrian
	-filter out tracks which are lost
	-calculate middle point of bounding box
	-makes new unique, scene-dependent ID (column 'metaId') since original dataset resets id for each scene
	-add scene name to column for visualization
	-output has columns=['trackId', 'frame', 'x', 'y', 'sceneId', 'metaId']

	before data needs to be in the following folder structure
	data/SDD/mode               mode can be 'train','val','test'
	|-bookstore_0
		|-annotations.txt
		|-reference.jpg
	|-scene_name
		|-...
	:param path: path to folder, default is 'data/SDD'
	:param mode: dataset split - options['train', 'test', 'val']
	:return: DataFrame containing all trajectories from dataset split
	'''
	assert mode in ['train', 'val', 'test']

	path = os.path.join(path, mode)
	scenes = os.listdir(path)
	SDD_cols = ['trackId', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label']
	data = []
	print('loading ' + mode + ' data')
	for scene in scenes:
		scene_path = os.path.join(path, scene, 'annotations.txt')
		scene_df = pd.read_csv(scene_path, header=0, names=SDD_cols, delimiter=' ')
		# Calculate center point of bounding box
		scene_df['x'] = (scene_df['xmax'] + scene_df['xmin']) / 2
		scene_df['y'] = (scene_df['ymax'] + scene_df['ymin']) / 2
		scene_df = scene_df[scene_df['label'] == 'Pedestrian']  # drop non-pedestrians
		scene_df = scene_df[scene_df['lost'] == 0]  # drop lost samples
		scene_df = scene_df.drop(columns=['xmin', 'xmax', 'ymin', 'ymax', 'occluded', 'generated', 'label', 'lost'])
		scene_df['sceneId'] = scene
		# new unique id by combining scene_id and track_id
		scene_df['rec&trackId'] = [recId + '_' + str(trackId).zfill(4) for recId, trackId in
								   zip(scene_df.sceneId, scene_df.trackId)]
		data.append(scene_df)
	data = pd.concat(data, ignore_index=True)
	rec_trackId2metaId = {}
	for i, j in enumerate(data['rec&trackId'].unique()):
		rec_trackId2metaId[j] = i
	data['metaId'] = [rec_trackId2metaId[i] for i in data['rec&trackId']]
	data = data.drop(columns=['rec&trackId'])
	return data


def mask_step(x, step):
	"""
	Create a mask to only contain the step-th element starting from the first element. Used to downsample
	"""
	mask = np.zeros_like(x)
	mask[::step] = 1
	return mask.astype(bool)


def downsample(df, step):
	"""
	Downsample data by the given step. Example, SDD is recorded in 30 fps, with step=30, the fps of the resulting
	df will become 1 fps. With step=12 the result will be 2.5 fps. It will do so individually for each unique
	pedestrian (metaId)
	:param df: pandas DataFrame - necessary to have column 'metaId'
	:param step: int - step size, similar to slicing-step param as in array[start:end:step]
	:return: pd.df - downsampled
	"""
	mask = df.groupby(['metaId'])['metaId'].transform(mask_step, step=step)
	return df[mask]


def filter_short_trajectories(df, threshold):
	"""
	Filter trajectories that are shorter in timesteps than the threshold
	:param df: pandas df with columns=['x', 'y', 'frame', 'trackId', 'sceneId', 'metaId']
	:param threshold: int - number of timesteps as threshold, only trajectories over threshold are kept
	:return: pd.df with trajectory length over threshold
	"""
	len_per_id = df.groupby(by='metaId', as_index=False).count()  # sequence-length for each unique pedestrian
	idx_over_thres = len_per_id[len_per_id['frame'] >= threshold]  # rows which are above threshold
	idx_over_thres = idx_over_thres['metaId'].unique()  # only get metaIdx with sequence-length longer than threshold
	df = df[df['metaId'].isin(idx_over_thres)]  # filter df to only contain long trajectories
	return df


def groupby_sliding_window(x, window_size, stride):
	x_len = len(x)
	n_chunk = (x_len - window_size) // stride + 1
	idx = []
	metaId = []
	for i in range(n_chunk):
		idx += list(range(i * stride, i * stride + window_size))
		metaId += ['{}_{}'.format(x.metaId.unique()[0], i)] * window_size
	# temp = x.iloc()[(i * stride):(i * stride + window_size)]
	# temp['new_metaId'] = '{}_{}'.format(x.metaId.unique()[0], i)
	# df = df.append(temp, ignore_index=True)
	df = x.iloc()[idx]
	df['newMetaId'] = metaId
	return df


def sliding_window(df, window_size, stride):
	"""
	Assumes downsampled df, chunks trajectories into chunks of length window_size. When stride < window_size then
	chunked trajectories are overlapping
	:param df: df
	:param window_size: sequence-length of one trajectory, mostly obs_len + pred_len
	:param stride: timesteps to move from one trajectory to the next one
	:return: df with chunked trajectories
	"""
	gb = df.groupby(['metaId'], as_index=False)
	df = gb.apply(groupby_sliding_window, window_size=window_size, stride=stride)
	df['metaId'] = pd.factorize(df['newMetaId'], sort=False)[0]
	df = df.drop(columns='newMetaId')
	df = df.reset_index(drop=True)
	return df


def split_at_fragment_lambda(x, frag_idx, gb_frag):
	""" Used only for split_fragmented() """
	metaId = x.metaId.iloc()[0]
	counter = 0
	if metaId in frag_idx:
		split_idx = gb_frag.groups[metaId]
		for split_id in split_idx:
			x.loc[split_id:, 'newMetaId'] = '{}_{}'.format(metaId, counter)
			counter += 1
	return x


def split_fragmented(df):
	"""
	Split trajectories when fragmented (defined as frame_{t+1} - frame_{t} > 1)
	Formally, this is done by changing the metaId at the fragmented frame and below
	:param df: DataFrame containing trajectories
	:return: df: DataFrame containing trajectories without fragments
	"""

	gb = df.groupby('metaId', as_index=False)
	# calculate frame_{t+1} - frame_{t} and fill NaN which occurs for the first frame of each track
	df['frame_diff'] = gb['frame'].diff().fillna(value=1.0).to_numpy()
	fragmented = df[df['frame_diff'] != 1.0]  # df containing all the first frames of fragmentation
	gb_frag = fragmented.groupby('metaId')  # helper for gb.apply
	frag_idx = fragmented.metaId.unique()  # helper for gb.apply
	df['newMetaId'] = df['metaId']  # temporary new metaId

	df = gb.apply(split_at_fragment_lambda, frag_idx, gb_frag)
	df['metaId'] = pd.factorize(df['newMetaId'], sort=False)[0]
	df = df.drop(columns='newMetaId')
	return df


def load_and_window_SDD(step, window_size, stride, path=None, mode='train', pickle_path=None):
	"""
	Helper function to aggregate loading and preprocessing in one function. Preprocessing contains:
	- Split fragmented trajectories
	- Downsample fps
	- Filter short trajectories below threshold=window_size
	- Sliding window with window_size and stride
	:param step (int): downsample factor, step=30 means 1fps and step=12 means 2.5fps on SDD
	:param window_size (int): Timesteps for one window
	:param stride (int): How many timesteps to stride in windowing. If stride=window_size then there is no overlap
	:param path (str): Path to SDD directory (not subdirectory, which is contained in mode)
	:param mode (str): Which dataset split, options=['train', 'val', 'test']
	:param pickle_path (str): Alternative to path+mode, if there already is a pickled version of the raw SDD as df
	:return pd.df: DataFrame containing the preprocessed data
	"""
	if pickle_path is not None:
		df = pd.read_pickle(pickle_path)
	else:
		df = load_SDD(path=path, mode=mode)
	df = split_fragmented(df)  # split track if frame is not continuous
	df = downsample(df, step=step)
	df = filter_short_trajectories(df, threshold=window_size)
	df = sliding_window(df, window_size=window_size, stride=stride)

	return df


def rot(df, image, k=1):
	'''
	Rotates image and coordinates counter-clockwise by k * 90° within image origin
	:param df: Pandas DataFrame with at least columns 'x' and 'y'
	:param image: PIL Image
	:param k: Number of times to rotate by 90°
	:return: Rotated Dataframe and image
	'''
	xy = df.copy()
	if image.ndim == 3:
		y0, x0, channels = image.shape
	else:
		y0, x0= image.shape

	xy.loc()[:, 'x'] = xy['x'] - x0 / 2
	xy.loc()[:, 'y'] = xy['y'] - y0 / 2
	c, s = np.cos(-k * np.pi / 2), np.sin(-k * np.pi / 2)
	R = np.array([[c, s], [-s, c]])
	xy.loc()[:, ['x', 'y']] = np.dot(xy[['x', 'y']], R)
	for i in range(k):
		image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

	if image.ndim == 3:
		y0, x0, channels = image.shape
	else:
		y0, x0= image.shape

	xy.loc()[:, 'x'] = xy['x'] + x0 / 2
	xy.loc()[:, 'y'] = xy['y'] + y0 / 2
	return xy, image


def fliplr(df, image):
	'''
	Flip image and coordinates horizontally
	:param df: Pandas DataFrame with at least columns 'x' and 'y'
	:param image: PIL Image
	:return: Flipped Dataframe and image
	'''
	xy = df.copy()
	if image.ndim == 3:
		y0, x0, channels = image.shape
	else:
		y0, x0= image.shape

	xy.loc()[:, 'x'] = xy['x'] - x0 / 2
	xy.loc()[:, 'y'] = xy['y'] - y0 / 2
	R = np.array([[-1, 0], [0, 1]])
	xy.loc()[:, ['x', 'y']] = np.dot(xy[['x', 'y']], R)
	image = cv2.flip(image, 1)

	if image.ndim == 3:
		y0, x0, channels = image.shape
	else:
		y0, x0= image.shape

	xy.loc()[:, 'x'] = xy['x'] + x0 / 2
	xy.loc()[:, 'y'] = xy['y'] + y0 / 2
	return xy, image


def augment_data(data, image_path='data/SDD/train', images={}, image_file='reference.jpg', seg_mask=False):
	'''
	Perform data augmentation
	:param data: Pandas df, needs x,y,metaId,sceneId columns
	:param image_path: example - 'data/SDD/val'
	:param images: dict with key being sceneId, value being PIL image
	:param image_file: str, image file name
	:param seg_mask: whether it's a segmentation mask or an image file
	:return:
	'''
	ks = [1, 2, 3]
	for scene in data.sceneId.unique():
		im_path = os.path.join(image_path, scene, image_file)
		if seg_mask:
			im = cv2.imread(im_path, 0)
		else:
			im = cv2.imread(im_path)
		images[scene] = im
	data_ = data.copy()  # data without rotation, used so rotated data can be appended to original df
	k2rot = {1: '_rot90', 2: '_rot180', 3: '_rot270'}
	for k in ks:
		metaId_max = data['metaId'].max()
		for scene in data_.sceneId.unique():
			im_path = os.path.join(image_path, scene, image_file)
			if seg_mask:
				im = cv2.imread(im_path, 0)
			else:
				im = cv2.imread(im_path)

			data_rot, im = rot(data_[data_.sceneId == scene], im, k)
			# image
			rot_angle = k2rot[k]
			images[scene + rot_angle] = im

			data_rot['sceneId'] = scene + rot_angle
			data_rot['metaId'] = data_rot['metaId'] + metaId_max + 1
			data = data.append(data_rot)

	metaId_max = data['metaId'].max()
	for scene in data.sceneId.unique():
		im = images[scene]
		data_flip, im_flip = fliplr(data[data.sceneId == scene], im)
		data_flip['sceneId'] = data_flip['sceneId'] + '_fliplr'
		data_flip['metaId'] = data_flip['metaId'] + metaId_max + 1
		data = data.append(data_flip)
		images[scene + '_fliplr'] = im_flip

	return data, images


def augment_eth_ucy_social(train_batches, train_scenes, train_masks, train_images):
	""" Augment ETH/UCY data that is preprocessed with social masks """
	# Rotate by 90°, 180°, 270°
	train_batches_aug = train_batches.copy()
	train_scenes_aug = train_scenes.copy()
	train_masks_aug = train_masks.copy()
	for scene in np.unique(train_scenes):
		image = train_images[scene].copy()
		for rot_times in range(1, 4):
			scene_trajectories = deepcopy(train_batches)
			scene_trajectories = scene_trajectories[train_scenes == scene]

			rot_angle = 90 * rot_times

			# Get middle point and calculate rotation matrix
			if image.ndim == 3:
				H, W, C = image.shape
			else:
				H, W = image.shape
			c, s = np.cos(-rot_times * np.pi / 2), np.sin(-rot_times * np.pi / 2)
			R = np.array([[c, s], [-s, c]])
			middle = np.array([W, H]) / 2

			# rotate image
			image_rot = image.copy()
			for _ in range(rot_times):
				image_rot = cv2.rotate(image_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)
			if image_rot.ndim == 3:
				H, W, C = image_rot.shape
			else:
				H, W = image_rot.shape
			# get new rotated middle point
			middle_rot = np.array([W, H]) / 2
			# perform transformation on trajectories
			for traj in scene_trajectories:
				# substract middle
				traj[:, :, 2:4] -= middle
				traj[:, :, 2:4] = np.dot(traj[:, :, 2:4], R)
				traj[:, :, 2:4] += middle_rot

			train_images[f'{scene}_{rot_angle}'] = image_rot
			train_batches_aug = np.append(train_batches_aug, scene_trajectories, axis=0)
			train_scenes_aug = np.append(train_scenes_aug,
										 np.array([f'{scene}_{rot_angle}'] * scene_trajectories.shape[0]), axis=0)
			train_masks_aug = np.append(train_masks_aug, train_masks[train_scenes == scene], axis=0)

	# Flip
	train_batches = deepcopy(train_batches_aug)
	train_scenes = deepcopy(train_scenes_aug)
	train_masks = deepcopy(train_masks_aug)
	for scene in np.unique(train_scenes):
		image = train_images[scene].copy()
		scene_trajectories = deepcopy(train_batches)
		scene_trajectories = scene_trajectories[train_scenes == scene]

		# Get middle point and calculate rotation matrix
		if image.ndim == 3:
			H, W, C = image.shape
		else:
			H, W = image.shape
		R = np.array([[-1, 0], [0, 1]])
		middle = np.array([W, H]) / 2

		# rotate image
		image_rot = image.copy()
		image_rot = cv2.flip(image_rot, 1)
		if image_rot.ndim == 3:
			H, W, C = image_rot.shape
		else:
			H, W = image_rot.shape
		# get new rotated middle point
		middle_rot = np.array([W, H]) / 2
		# perform transformation on trajectories
		for traj in scene_trajectories:
			# substract middle
			traj[:, :, 2:4] -= middle
			traj[:, :, 2:4] = np.dot(traj[:, :, 2:4], R)
			traj[:, :, 2:4] += middle_rot

		train_images[f'{scene}_flip'] = image_rot
		train_batches_aug = np.append(train_batches_aug, scene_trajectories, axis=0)
		train_scenes_aug = np.append(train_scenes_aug, np.array([f'{scene}_flip'] * scene_trajectories.shape[0]),
									 axis=0)
		train_masks_aug = np.append(train_masks_aug, train_masks[train_scenes == scene], axis=0)
	return train_batches_aug, train_scenes_aug, train_masks_aug


def resize_and_pad_image(images, size, pad=2019):
	""" Resize image to desired size and pad image to make it square shaped and ratio of images still is the same, as
	images all have different sizes.
	"""
	for key, im in images.items():
		H, W, C = im.shape
		im = cv2.copyMakeBorder(im, 0, pad - H, 0, pad - W, cv2.BORDER_CONSTANT)
		im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
		images[key] = im


def create_images_dict(data, image_path, image_file='reference.jpg'):
	images = {}
	for scene in data.sceneId.unique():
		if image_file == 'oracle.png':
			im = cv2.imread(os.path.join(image_path, scene, image_file), 0)
		else:
			im = cv2.imread(os.path.join(image_path, scene, image_file))
		images[scene] = im
	return images


def load_images(scenes, image_path, image_file='reference.jpg'):
	images = {}
	if type(scenes) is list:
		scenes = set(scenes)
	for scene in scenes:
		if image_file == 'oracle.png':
			im = cv2.imread(os.path.join(image_path, scene, image_file), 0)
		else:
			im = cv2.imread(os.path.join(image_path, scene, image_file))
		images[scene] = im
	return images


def read_trajnet(mode='train'):
	root = 'data/SDD_trajnet/'
	path = os.path.join(root, mode)

	fp = os.listdir(path)
	df_list = []
	for file in fp:
		name = file.split('.txt')[0]

		df = pd.read_csv(os.path.join(path, file), sep=' ', names=['frame', 'trackId', 'x', 'y'])
		df['sceneId'] = name
		df_list.append(df)

	df = pd.concat(df_list, ignore_index=True)
	df['metaId'] = [recId + '_' + str(trackId).zfill(4) for recId, trackId in zip(df.sceneId, df.trackId)]
	df['metaId'] = pd.factorize(df['metaId'], sort=False)[0]
	return df


def load_inD(path='data/inD/', scenes=[1], recordings=None):
	'''
	Loads data from inD Dataset. Makes the following preprocessing:
	-filter out unnecessary columns
	-filter out non-pedestrian
	-makes new unique ID (column 'metaId') since original dataset resets id for each scene
	-add scene name to column for visualization
	-output has columns=['trackId', 'frame', 'x', 'y', 'sceneId', 'metaId']

	data needs to be in the following folder structure
	data/inD/*.csv

	:param path: str - path to folder, default is 'data/inD'
	:param scenes: list of integers - scenes to load
	:param recordings: list of strings - alternative to scenes, load specified recordings instead, overwrites scenes
	:return: DataFrame containing all trajectories from split
	'''

	scene2rec = {1: ['00', '01', '02', '03', '04', '05', '06'],
				 2: ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17'],
				 3: ['18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29'],
				 4: ['30', '31', '32']}

	rec_to_load = []
	for scene in scenes:
		rec_to_load.extend(scene2rec[scene])
	if recordings is not None:
		rec_to_load = recordings
	data = []
	for rec in rec_to_load:
		# load csv
		track = pd.read_csv(os.path.join(path, '{}_tracks.csv'.format(rec)))
		track = track.drop(columns=['trackLifetime', 'heading', 'width', 'length', 'xVelocity', 'yVelocity',
									'xAcceleration', 'yAcceleration', 'lonVelocity', 'latVelocity',
									'lonAcceleration', 'latAcceleration'])
		track_meta = pd.read_csv(os.path.join(path, '{}_tracksMeta.csv'.format(rec)))

		# Filter non-pedestrians
		pedestrians = track_meta[track_meta['class'] == 'pedestrian']
		track = track[track['trackId'].isin(pedestrians['trackId'])]

		track['rec&trackId'] = [str(recId) + '_' + str(trackId).zfill(6) for recId, trackId in
								zip(track.recordingId, track.trackId)]
		track['sceneId'] = rec
		track['yCenter'] = -track['yCenter']
		data.append(track)

	data = pd.concat(data, ignore_index=True)

	rec_trackId2metaId = {}
	for i, j in enumerate(data['rec&trackId'].unique()):
		rec_trackId2metaId[j] = i
	data['metaId'] = [rec_trackId2metaId[i] for i in data['rec&trackId']]
	data = data.drop(columns=['rec&trackId', 'recordingId'])
	data = data.rename(columns={'xCenter': 'x', 'yCenter': 'y'})

	cols_order = ['trackId', 'frame', 'x', 'y', 'sceneId', 'metaId']
	data = data.reindex(columns=cols_order)
	return data


def load_and_window_inD(step, window_size, stride, scenes=[1,2,3,4], pickle=False):
	"""
	Helper function to aggregate loading and preprocessing in one function. Preprocessing contains:
	- Split fragmented trajectories
	- Downsample fps
	- Filter short trajectories below threshold=window_size
	- Sliding window with window_size and stride
	:param step (int): downsample factor, step=30 means 1fps and step=12 means 2.5fps on SDD
	:param window_size (int): Timesteps for one window
	:param stride (int): How many timesteps to stride in windowing. If stride=window_size then there is no overlap
	:param scenes (list of int): Which scenes to load, inD has 4 scenes
	:param pickle (Bool): If True, load pickle instead of csv
	:return pd.df: DataFrame containing the preprocessed data
	"""
	df = load_inD(path='data/inD/', scenes=scenes, recordings=None)
	df = downsample(df, step=step)
	df = filter_short_trajectories(df, threshold=window_size)
	df = sliding_window(df, window_size=window_size, stride=stride)

	return df

# train-scenes are 2, 3, 4
# test scene is 1

# df = load_and_window_inD(step=25, window_size=35, stride=35, scenes=[1], pickle=False)
# df.to_pickle('test.pickle')