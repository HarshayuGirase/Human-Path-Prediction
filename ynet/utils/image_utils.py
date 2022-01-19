import numpy as np
import torch
import cv2
import torch.nn.functional as F

def gkern(kernlen=31, nsig=4):
	"""	creates gaussian kernel with side length l and a sigma of sig """
	ax = np.linspace(-(kernlen - 1) / 2., (kernlen - 1) / 2., kernlen)
	xx, yy = np.meshgrid(ax, ax)
	kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(nsig))
	return kernel / np.sum(kernel)


def create_gaussian_heatmap_template(size, kernlen=81, nsig=4, normalize=True):
	""" Create a big gaussian heatmap template to later get patches out """
	template = np.zeros([size, size])
	kernel = gkern(kernlen=kernlen, nsig=nsig)
	m = kernel.shape[0]
	x_low = template.shape[1] // 2 - int(np.floor(m / 2))
	x_up = template.shape[1] // 2 + int(np.ceil(m / 2))
	y_low = template.shape[0] // 2 - int(np.floor(m / 2))
	y_up = template.shape[0] // 2 + int(np.ceil(m / 2))
	template[y_low:y_up, x_low:x_up] = kernel
	if normalize:
		template = template / template.max()
	return template


def create_dist_mat(size, normalize=True):
	""" Create a big distance matrix template to later get patches out """
	middle = size // 2
	dist_mat = np.linalg.norm(np.indices([size, size]) - np.array([middle, middle])[:,None,None], axis=0)
	if normalize:
		dist_mat = dist_mat / dist_mat.max() * 2
	return dist_mat


def get_patch(template, traj, H, W):
	x = np.round(traj[:,0]).astype('int')
	y = np.round(traj[:,1]).astype('int')

	x_low = template.shape[1] // 2 - x
	x_up = template.shape[1] // 2 + W - x
	y_low = template.shape[0] // 2 - y
	y_up = template.shape[0] // 2 + H - y

	patch = [template[y_l:y_u, x_l:x_u] for x_l, x_u, y_l, y_u in zip(x_low, x_up, y_low, y_up)]

	return patch


def preprocess_image_for_segmentation(images, encoder='resnet101', encoder_weights='imagenet', seg_mask=False, classes=6):
	""" Preprocess image for pretrained semantic segmentation, input is dictionary containing images
	In case input is segmentation map, then it will create one-hot-encoding from discrete values"""
	import segmentation_models_pytorch as smp

	preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

	for key, im in images.items():
		if seg_mask:
			im = [(im == v) for v in range(classes)]
			im = np.stack(im, axis=-1)  # .astype('int16')
		else:
			im = preprocessing_fn(im)
		im = im.transpose(2, 0, 1).astype('float32')
		im = torch.Tensor(im)
		images[key] = im


def resize(images, factor, seg_mask=False):
	for key, image in images.items():
		if seg_mask:
			images[key] = cv2.resize(image, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
		else:
			images[key] = cv2.resize(image, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def pad(images, division_factor=32):
	""" Pad image so that it can be divided by division_factor, as many architectures such as UNet needs a specific size
	at it's bottlenet layer"""
	for key, im in images.items():
		if im.ndim == 3:
			H, W, C = im.shape
		else:
			H, W = im.shape
		H_new = int(np.ceil(H / division_factor) * division_factor)
		W_new = int(np.ceil(W / division_factor) * division_factor)
		im = cv2.copyMakeBorder(im, 0, H_new - H, 0, W_new - W, cv2.BORDER_CONSTANT)
		images[key] = im


def sampling(probability_map, num_samples, rel_threshold=None, replacement=False):
	# new view that has shape=[batch*timestep, H*W]
	prob_map = probability_map.view(probability_map.size(0) * probability_map.size(1), -1)
	if rel_threshold is not None:
		thresh_values = prob_map.max(dim=1)[0].unsqueeze(1).expand(-1, prob_map.size(1))
		mask = prob_map < thresh_values * rel_threshold
		prob_map = prob_map * (~mask).int()
		prob_map = prob_map / prob_map.sum()

	# samples.shape=[batch*timestep, num_samples]
	samples = torch.multinomial(prob_map, num_samples=num_samples, replacement=replacement)
	# samples.shape=[batch, timestep, num_samples]

	# unravel sampled idx into coordinates of shape [batch, time, sample, 2]
	samples = samples.view(probability_map.size(0), probability_map.size(1), -1)
	idx = samples.unsqueeze(3)
	preds = idx.repeat(1, 1, 1, 2).float()
	preds[:, :, :, 0] = (preds[:, :, :, 0]) % probability_map.size(3)
	preds[:, :, :, 1] = torch.floor((preds[:, :, :, 1]) / probability_map.size(3))

	return preds


def image2world(image_coords, scene, homo_mat, resize):
	"""
	Transform trajectories of one scene from image_coordinates to world_coordinates
	:param image_coords: torch.Tensor, shape=[num_person, (optional: num_samples), timesteps, xy]
	:param scene: string indicating current scene, options=['eth', 'hotel', 'student01', 'student03', 'zara1', 'zara2']
	:param homo_mat: dict, key is scene, value is torch.Tensor containing homography matrix (data/eth_ucy/scene_name.H)
	:param resize: float, resize factor
	:return: trajectories in world_coordinates
	"""
	traj_image2world = image_coords.clone()
	if traj_image2world.dim() == 4:
		traj_image2world = traj_image2world.reshape(-1, image_coords.shape[2], 2)
	if scene in ['eth', 'hotel']:
		# eth and hotel have different coordinate system than ucy data
		traj_image2world[:, :, [0, 1]] = traj_image2world[:, :, [1, 0]]
	traj_image2world = traj_image2world / resize
	traj_image2world = F.pad(input=traj_image2world, pad=(0, 1, 0, 0), mode='constant', value=1)
	traj_image2world = traj_image2world.reshape(-1, 3)
	traj_image2world = torch.matmul(homo_mat[scene], traj_image2world.T).T
	traj_image2world = traj_image2world / traj_image2world[:, 2:]
	traj_image2world = traj_image2world[:, :2]
	traj_image2world = traj_image2world.view_as(image_coords)
	return traj_image2world
