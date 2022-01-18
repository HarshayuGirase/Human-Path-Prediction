import numpy as np
import pandas as pd
import os


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


def load_inD(path='inD-dataset-v1.0/data', scenes=[1], recordings=None):
	'''
	Loads data from inD Dataset. Makes the following preprocessing:
	-filter out unnecessary columns
	-filter out non-pedestrian
	-makes new unique ID (column 'metaId') since original dataset resets id for each scene
	-add scene name to column for visualization
	-output has columns=['trackId', 'frame', 'x', 'y', 'sceneId', 'metaId']

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

		# Filter all trajectories outside the scene frame, ie negative values
		track = track[(track['yCenter'] >= 0) & (track['xCenter'] >= 0)]

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


def load_and_window_inD(step, window_size, stride, scenes=[1,2,3,4], path='inD-dataset-v1.0/data'):
	"""
	Helper function to aggregate loading and preprocessing in one function. Preprocessing contains:
	- Split fragmented trajectories
	- Downsample fps
	- Filter short trajectories below threshold=window_size
	- Sliding window with window_size and stride
	:param step (int): downsample factor, step=25 means 1fps on inD
	:param window_size (int): Timesteps for one window
	:param stride (int): How many timesteps to stride in windowing. If stride=window_size then there is no overlap
	:param scenes (list of int): Which scenes to load, inD has 4 scenes
	:return pd.df: DataFrame containing the preprocessed data
	"""
	rec2scene = {'00': 'scene1', '01': 'scene1', '02': 'scene1', '03': 'scene1', '04': 'scene1', '05': 'scene1',
				 '06': 'scene1',
				 '07': 'scene2', '08': 'scene2', '09': 'scene2', '10': 'scene2', '11': 'scene2', '12': 'scene2',
				 '13': 'scene2', '14': 'scene2', '15': 'scene2', '16': 'scene2', '17': 'scene2',
				 '18': 'scene3', '19': 'scene3', '20': 'scene3', '21': 'scene3', '22': 'scene3', '23': 'scene3',
				 '24': 'scene3', '25': 'scene3', '26': 'scene3', '27': 'scene3', '28': 'scene3', '29': 'scene3',
				 '30': 'scene4', '31': 'scene4', '32': 'scene4'}

	df = load_inD(path=path, scenes=scenes, recordings=None)
	# inD is already perfectly continuous
	df = downsample(df, step=step)
	df = filter_short_trajectories(df, threshold=window_size)
	df = sliding_window(df, window_size=window_size, stride=stride)
	df['recId'] = df['sceneId'].copy()
	df['sceneId'] = df['recId'].map(rec2scene)

	# To scale inD x and y values into pixel coordinates, one has to divide scene 1 with 0.00814 * 12 and the others with 0.0127 * 12
	# Values from the XX_recordingMeta.csv "orthoPxToMeter" * 12
	df.x /= np.where(df.sceneId == 'scene1', 0.0127 * 12, 0.00814 * 12)
	df.y /= np.where(df.sceneId == 'scene1', 0.0127 * 12, 0.00814 * 12)

	return df


if __name__ == "__main__":
	df_test = load_and_window_inD(step=25, window_size=35, stride=35, scenes=[1], path='inD-dataset-v1.0/data')
	df_test.to_pickle('inD_test.pickle')

	df_train = load_and_window_inD(step=25, window_size=35, stride=35, scenes=[2, 3, 4], path='inD-dataset-v1.0/data')
	df_train.to_pickle('inD_train.pickle')
