import math

import numpy as np
import wandb

from jax_marl.viz.window import Window
import jax_marl.viz.grid_rendering as rendering
from jax_marl.environments.overcooked_environment.common import OBJECT_TO_INDEX, COLOR_TO_INDEX, COLORS


INDEX_TO_COLOR = [k for k,v in COLOR_TO_INDEX.items()]
TILE_PIXELS = 32

COLOR_TO_AGENT_INDEX = {0:0, 2:1} # Hardcoded. Red is first, blue is second

class OvercookedVisualizer:
	"""
	Manages a window and renders contents of EnvState instances to it.
	"""
	tile_cache = {}

	def __init__(self):
		self.window = None

	def _lazy_init_window(self):
		if self.window is None:
			self.window = Window('minimax')

	def show(self, block=False):
		self._lazy_init_window()
		self.window.show(block=block)

	def render(self, agent_view_size, state, highlight=True, tile_size=TILE_PIXELS):
		"""Method for rendering the state in a window. Esp. useful for interactive mode."""
		return self._render_state(agent_view_size, state, highlight, tile_size)

	def animate(self, state_seq, agent_view_size, task_idx, task_name, exp_dir):
		"""Animate a gif give a state sequence and save if to file."""
		import imageio

		padding = agent_view_size - 2  # show

		def get_frame(state):
			grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
			# Render the state
			frame = OvercookedVisualizer._render_grid(
				grid,
				tile_size=TILE_PIXELS,
				highlight_mask=None,
				agent_dir_idx=state.agent_dir_idx,
				agent_inv=state.agent_inv
			)
			return frame

		frame_seq =[get_frame(state.env_state) for state in state_seq]
		file_name = f"task_{task_idx}_{task_name}"
		file_path = f"{exp_dir}/{file_name}.gif"

		imageio.mimsave(file_path, frame_seq, 'GIF')
		wandb.log({file_name: wandb.Video(file_path, format="gif")})


	def render_grid(self, grid, tile_size=TILE_PIXELS, k_rot90=0, agent_dir_idx=None):
		self._lazy_init_window()

		img = OvercookedVisualizer._render_grid(
				grid,
				tile_size,
				highlight_mask=None,
				agent_dir_idx=agent_dir_idx,
			)
		# img = np.transpose(img, axes=(1,0,2))
		if k_rot90 > 0:
			img = np.rot90(img, k=k_rot90)

		self.window.show_img(img)

	def _render_state(self, agent_view_size, state, highlight=True, tile_size=TILE_PIXELS):
		"""
		Render the state
		"""
		self._lazy_init_window()

		padding = agent_view_size-2 # show
		grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
		grid_offset = np.array([1,1])
		h, w = grid.shape[:2]
		# === Compute highlight mask
		highlight_mask = np.zeros(shape=(h,w), dtype=bool)

		if highlight:
			# TODO: Fix this for multiple agents
			f_vec = state.agent_dir
			r_vec = np.array([-f_vec[1], f_vec[0]])

			fwd_bound1 = state.agent_pos
			fwd_bound2 = state.agent_pos + state.agent_dir*(agent_view_size-1)
			side_bound1 = state.agent_pos - r_vec*(agent_view_size//2)
			side_bound2 = state.agent_pos + r_vec*(agent_view_size//2)

			min_bound = np.min(np.stack([
						fwd_bound1,
						fwd_bound2,
						side_bound1,
						side_bound2]) + grid_offset, 0)

			min_y = min(max(min_bound[1], 0), highlight_mask.shape[0]-1)
			min_x = min(max(min_bound[0],0), highlight_mask.shape[1]-1)

			max_y = min(max(min_bound[1]+agent_view_size, 0), highlight_mask.shape[0]-1)
			max_x = min(max(min_bound[0]+agent_view_size, 0), highlight_mask.shape[1]-1)

			highlight_mask[min_y:max_y,min_x:max_x] = True

		# Render the whole grid
		img = OvercookedVisualizer._render_grid(
			grid,
			tile_size,
			highlight_mask=highlight_mask if highlight else None,
			agent_dir_idx=state.agent_dir_idx,
			agent_inv=state.agent_inv
		)
		self.window.show_img(img)

	@classmethod
	def _render_obj(
		cls,
		obj,
		img):
		# Render each kind of object
		obj_type = obj[0]
		color = INDEX_TO_COLOR[obj[1]]

		if obj_type == OBJECT_TO_INDEX['wall']:
			rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS[color])
		elif obj_type == OBJECT_TO_INDEX['goal']:
			rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
			rendering.fill_coords(img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS[color])
		elif obj_type == OBJECT_TO_INDEX['agent']:
			agent_dir_idx = obj[2]
			tri_fn = rendering.point_in_triangle(
				(0.12, 0.19),
				(0.87, 0.50),
				(0.12, 0.81),
			)
			# tri_fn = rendering.rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*agent_dir_idx)

			# A bit hacky, but needed so that actions order matches the one of Overcooked-AI
			direction_reording = [3,1,0,2]
			direction = direction_reording[agent_dir_idx]
			tri_fn = rendering.rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*direction)
			
			rendering.fill_coords(img, tri_fn, COLORS[color])
		elif obj_type == OBJECT_TO_INDEX['empty']:
			rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS[color])
		elif obj_type == OBJECT_TO_INDEX['onion_pile']:
			rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
			onion_fns = [rendering.point_in_circle(*coord, 0.15) for coord in [(0.5, 0.15), (0.3, 0.4), (0.8, 0.35),
																			  (0.4, 0.8), (0.75, 0.75)]]
			[rendering.fill_coords(img, onion_fn, COLORS[color]) for onion_fn in onion_fns]
		elif obj_type == OBJECT_TO_INDEX['onion']:
			rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
			onion_fn = rendering.point_in_circle(0.5, 0.5, 0.15)
			rendering.fill_coords(img, onion_fn, COLORS[color])
		elif obj_type == OBJECT_TO_INDEX['plate_pile']:
			rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
			plate_fns = [rendering.point_in_circle(*coord, 0.2) for coord in [(0.3, 0.3), (0.75, 0.42),
																			  (0.4, 0.75)]]
			[rendering.fill_coords(img, plate_fn, COLORS[color]) for plate_fn in plate_fns]
		elif obj_type == OBJECT_TO_INDEX['plate']:
			rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
			plate_fn = rendering.point_in_circle(0.5, 0.5, 0.2)
			rendering.fill_coords(img, plate_fn, COLORS[color])
		elif obj_type == OBJECT_TO_INDEX['dish']:
			rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
			plate_fn = rendering.point_in_circle(0.5, 0.5, 0.2)
			rendering.fill_coords(img, plate_fn, COLORS[color])
			onion_fn = rendering.point_in_circle(0.5, 0.5, 0.13)
			rendering.fill_coords(img, onion_fn, COLORS["orange"])
		elif obj_type == OBJECT_TO_INDEX['pot']:
			OvercookedVisualizer._render_pot(obj, img)
			# rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
			# pot_fns = [rendering.point_in_rect(0.1, 0.9, 0.3, 0.9),
			# 		   rendering.point_in_rect(0.1, 0.9, 0.20, 0.23),
			# 		   rendering.point_in_rect(0.4, 0.6, 0.15, 0.20),]
			# [rendering.fill_coords(img, pot_fn, COLORS[color]) for pot_fn in pot_fns]
		else:
			raise ValueError(f'Rendering object at index {obj_type} is currently unsupported.')

	@classmethod
	def _render_pot(
		cls,
		obj,
		img):
		pot_status = obj[-1]
		num_onions = np.max([23-pot_status, 0])
		is_cooking = np.array((pot_status < 20) * (pot_status > 0))
		is_done = np.array(pot_status == 0)

		pot_fn = rendering.point_in_rect(0.1, 0.9, 0.33, 0.9)
		lid_fn = rendering.point_in_rect(0.1, 0.9, 0.21, 0.25)
		handle_fn = rendering.point_in_rect(0.4, 0.6, 0.16, 0.21)

		rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])

		# Render onions in pot
		if num_onions > 0 and not is_done:
			onion_fns = [rendering.point_in_circle(*coord, 0.13) for coord in [(0.23, 0.33), (0.77, 0.33), (0.50, 0.33)]]
			onion_fns = onion_fns[:num_onions]
			[rendering.fill_coords(img, onion_fn, COLORS["yellow"]) for onion_fn in onion_fns]
			if not is_cooking:
				lid_fn = rendering.rotate_fn(lid_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)
				handle_fn =rendering.rotate_fn(handle_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)

		# Render done soup
		if is_done:
			soup_fn = rendering.point_in_rect(0.12, 0.88, 0.23, 0.35)
			rendering.fill_coords(img, soup_fn, COLORS["orange"])

		# Render the pot itself
		pot_fns = [pot_fn, lid_fn, handle_fn]
		[rendering.fill_coords(img, pot_fn, COLORS["black"]) for pot_fn in pot_fns]

		# Render progress bar
		if is_cooking:
			progress_fn = rendering.point_in_rect(0.1, 0.9-(0.9-0.1)/20*pot_status, 0.83, 0.88)
			rendering.fill_coords(img, progress_fn, COLORS["green"])


	@classmethod
	def _render_inv(
		cls,
		obj,
		img):
		# Render each kind of object
		obj_type = obj[0]
		if obj_type == OBJECT_TO_INDEX['empty']:
			pass
		elif obj_type == OBJECT_TO_INDEX['onion']:
			onion_fn = rendering.point_in_circle(0.75, 0.75, 0.15)
			rendering.fill_coords(img, onion_fn, COLORS["yellow"])
		elif obj_type == OBJECT_TO_INDEX['plate']:
			plate_fn = rendering.point_in_circle(0.75, 0.75, 0.2)
			rendering.fill_coords(img, plate_fn, COLORS["white"])
		elif obj_type == OBJECT_TO_INDEX['dish']:
			plate_fn = rendering.point_in_circle(0.75, 0.75, 0.2)
			rendering.fill_coords(img, plate_fn, COLORS["white"])
			onion_fn = rendering.point_in_circle(0.75, 0.75, 0.13)
			rendering.fill_coords(img, onion_fn, COLORS["orange"])
		else:
			raise ValueError(f'Rendering object at index {obj_type} is currently unsupported.')

	@classmethod
	def _render_tile(
		cls,
		obj,
		highlight=False,
		agent_dir_idx=None,
		agent_inv=None,
		tile_size=TILE_PIXELS,
		subdivs=3
	):
		"""
		Render a tile and cache the result
		"""
		# Hash map lookup key for the cache
		if obj is not None and obj[0] == OBJECT_TO_INDEX['agent']:
			# Get inventory of this specific agent
			if agent_inv is not None:
				color_idx = obj[1]
				agent_inv = agent_inv[COLOR_TO_AGENT_INDEX[color_idx]]
				agent_inv = np.array([agent_inv, -1, 0])

			if agent_dir_idx is not None:
				obj = np.array(obj)

				# TODO: Fix this for multiagents. Currently the orientation of other agents is wrong
				if len(agent_dir_idx) == 1:
					# Hacky way of making agent views orientations consistent with global view
					obj[-1] = agent_dir_idx[0]

		no_object = obj is None or (
			obj[0] in [OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['unseen']] \
			and obj[2] == 0
		)

		if not no_object:
			if agent_inv is not None and obj[0] == OBJECT_TO_INDEX['agent']:
				key = (*obj, *agent_inv, highlight, tile_size)
			else:
				key = (*obj, highlight, tile_size)
		else:
			key = (obj, highlight, tile_size)

		if key in cls.tile_cache:
			return cls.tile_cache[key]

		img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

		# Draw the grid lines (top and left edges)
		rendering.fill_coords(img, rendering.point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
		rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

		if not no_object:
			OvercookedVisualizer._render_obj(obj, img)
			# render inventory
			if agent_inv is not None and obj[0] == OBJECT_TO_INDEX['agent']:
				OvercookedVisualizer._render_inv(agent_inv, img)


		if highlight:
			rendering.highlight_img(img)

		# Downsample the image to perform supersampling/anti-aliasing
		img = rendering.downsample(img, subdivs)

		# Cache the rendered tile
		cls.tile_cache[key] = img

		return img

	@classmethod
	def _render_grid(
		cls,
		grid,
		tile_size=TILE_PIXELS,
		highlight_mask=None,
		agent_dir_idx=None,
		agent_inv=None):
		if highlight_mask is None:
			highlight_mask = np.zeros(shape=grid.shape[:2], dtype=bool)

		# Compute the total grid size in pixels
		width_px = grid.shape[1]*tile_size
		height_px = grid.shape[0]*tile_size

		img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

		# Render the grid
		for y in range(grid.shape[0]):
			for x in range(grid.shape[1]):		
				obj = grid[y,x,:]
				if obj[0] in [OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['unseen']] \
					and obj[2] == 0:
					obj = None

				tile_img = OvercookedVisualizer._render_tile(
					obj,
					highlight=highlight_mask[y, x],
					tile_size=tile_size,
					agent_dir_idx=agent_dir_idx,
					agent_inv=agent_inv,
				)

				ymin = y*tile_size
				ymax = (y+1)*tile_size
				xmin = x*tile_size
				xmax = (x+1)*tile_size
				img[ymin:ymax, xmin:xmax, :] = tile_img

		return img

	def close(self):
		self.window.close()