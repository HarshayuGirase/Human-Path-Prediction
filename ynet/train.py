import torch
import torch.nn as nn
from utils.image_utils import get_patch, image2world


def train(model, train_loader, train_images, e, obs_len, pred_len, batch_size, params, gt_template, device, input_template, optimizer, criterion, dataset_name, homo_mat):
	"""
	Run training for one epoch

	:param model: torch model
	:param train_loader: torch dataloader
	:param train_images: dict with keys: scene_name value: preprocessed image as torch.Tensor
	:param e: epoch number
	:param params: dict of hyperparameters
	:param gt_template:  precalculated Gaussian heatmap template as torch.Tensor
	:return: train_ADE, train_FDE, train_loss for one epoch
	"""
	train_loss = 0
	train_ADE = []
	train_FDE = []
	model.train()
	counter = 0
	# outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once
	for batch, (trajectory, meta, scene) in enumerate(train_loader):
		# Stop training after 25 batches to increase evaluation frequency
		if dataset_name == 'sdd' and obs_len == 8 and batch > 25:
			break

		# TODO Delete
		if dataset_name == 'eth':
			print(counter)
			counter += batch_size
			# Break after certain number of batches to approximate evaluation, else one epoch takes really long
			if counter > 30: #TODO Delete
				break


		# Get scene image and apply semantic segmentation
		if e < params['unfreeze']:  # before unfreeze only need to do semantic segmentation once
			model.eval()
			scene_image = train_images[scene].to(device).unsqueeze(0)
			scene_image = model.segmentation(scene_image)
			model.train()

		# inner loop, for each trajectory in the scene
		for i in range(0, len(trajectory), batch_size):
			if e >= params['unfreeze']:
				scene_image = train_images[scene].to(device).unsqueeze(0)
				scene_image = model.segmentation(scene_image)

			# Create Heatmaps for past and ground-truth future trajectories
			_, _, H, W = scene_image.shape  # image shape

			observed = trajectory[i:i+batch_size, :obs_len, :].reshape(-1, 2).cpu().numpy()
			observed_map = get_patch(input_template, observed, H, W)
			observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])

			gt_future = trajectory[i:i + batch_size, obs_len:].to(device)
			gt_future_map = get_patch(gt_template, gt_future.reshape(-1, 2).cpu().numpy(), H, W)
			gt_future_map = torch.stack(gt_future_map).reshape([-1, pred_len, H, W])

			gt_waypoints = gt_future[:, params['waypoints']]
			gt_waypoint_map = get_patch(input_template, gt_waypoints.reshape(-1, 2).cpu().numpy(), H, W)
			gt_waypoint_map = torch.stack(gt_waypoint_map).reshape([-1, gt_waypoints.shape[1], H, W])

			# Concatenate heatmap and semantic map
			semantic_map = scene_image.expand(observed_map.shape[0], -1, -1, -1)  # expand to match heatmap size
			feature_input = torch.cat([semantic_map, observed_map], dim=1)

			# Forward pass
			# Calculate features
			features = model.pred_features(feature_input)

			# Predict goal and waypoint probability distribution
			pred_goal_map = model.pred_goal(features)
			goal_loss = criterion(pred_goal_map, gt_future_map) * params['loss_scale']  # BCEWithLogitsLoss

			# Prepare (downsample) ground-truth goal and trajectory heatmap representation for conditioning trajectory decoder
			gt_waypoints_maps_downsampled = [nn.AvgPool2d(kernel_size=2**i, stride=2**i)(gt_waypoint_map) for i in range(1, len(features))]
			gt_waypoints_maps_downsampled = [gt_waypoint_map] + gt_waypoints_maps_downsampled

			# Predict trajectory distribution conditioned on goal and waypoints
			traj_input = [torch.cat([feature, goal], dim=1) for feature, goal in zip(features, gt_waypoints_maps_downsampled)]
			pred_traj_map = model.pred_traj(traj_input)
			traj_loss = criterion(pred_traj_map, gt_future_map) * params['loss_scale']  # BCEWithLogitsLoss

			# Backprop
			loss = goal_loss + traj_loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			with torch.no_grad():
				train_loss += loss
				# Evaluate using Softargmax, not a very exact evaluation but a lot faster than full prediction
				pred_traj = model.softargmax(pred_traj_map)
				pred_goal = model.softargmax(pred_goal_map[:, -1:])

				# converts ETH/UCY pixel coordinates back into world-coordinates
				# if dataset_name == 'eth':
				# 	pred_goal = image2world(pred_goal, scene, homo_mat, params)
				# 	pred_traj = image2world(pred_traj, scene, homo_mat, params)
				# 	gt_future = image2world(gt_future, scene, homo_mat, params)

				train_ADE.append(((((gt_future - pred_traj) / params['resize']) ** 2).sum(dim=2) ** 0.5).mean(dim=1))
				train_FDE.append(((((gt_future[:, -1:] - pred_goal[:, -1:]) / params['resize']) ** 2).sum(dim=2) ** 0.5).mean(dim=1))

	train_ADE = torch.cat(train_ADE).mean()
	train_FDE = torch.cat(train_FDE).mean()

	return train_ADE.item(), train_FDE.item(), train_loss.item()
