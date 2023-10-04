import  os.path
import  numpy as np
import random
from datetime import datetime
import sys

n_air=1.
n_Si=3.567390909090909
n_sub=1.45

EPSILON_0 = 8.85418782e-12        # vacuum permittivity
MU_0 = 1.25663706e-6              # vacuum permeability
C_0 = 1 / np.sqrt(EPSILON_0 * MU_0)  # speed of light in vacuum
ETA_0 = np.sqrt(MU_0 / EPSILON_0)    # vacuum impedance
Q_e = 1.602176634e-19             # funamental charge

class SimulationDataset(object):
	def __init__(self, data_folder, img_file_name=None, field_file_name=None, sources_file_name=None, crops_per_idx = 6, crop_dx=64, crop_dy=64, boundary_size=200, total_sample_number = None, transform = None, save_folder=None, dL=6.25e-9, wl=1050e-9):

		# data_folder = config.get('data_folder', '/scratch/users/chenkaim/data/') # for sherlock
		#self.input_imgs = np.load(data_folder+'/'+'grating_pattern_UNet_64_256_52thick_8bar_with_sub.npy', mmap_mode='r') 
		self.input_imgs = np.load(data_folder+'/'+img_file_name, mmap_mode='r')[:,:,:] 
		#self.input_imgs = np.load(data_folder+'/100k_test_3k_input_imgs.npy', mmap_mode='r') 	

		self.input_imgs = self.input_imgs[:, :, :].astype(np.float32, copy=False)
		#self.input_imgs = np.squeeze(np.swapaxes(self.input_imgs, 1, 3));
		# sub = np.ones((self.input_imgs.shape[0],1,5,256))*(1.45-1.)/(3.567390909090909-1.)
		# self.input_imgs=np.concatenate([sub,self.input_imgs],axis=2).astype(np.float32, copy=False);
		print("input_imgs.shape: ", self.input_imgs.shape, self.input_imgs.dtype)
		
		#self.Hy_forward = np.load(data_folder+'/'+'Hy_out_forward_RI.npy', mmap_mode='r');
		self.Hy_forward = np.load(data_folder+'/'+field_file_name, mmap_mode='r');
		print("Hy_forward.shape: ", self.Hy_forward.shape, self.Hy_forward.dtype)
		self.fields = self.Hy_forward;

		# self.sources = np.load(data_folder+'/'+sources_file_name, mmap_mode='r');
		# print("Sources.shape: ", self.sources.shape, self.sources.dtype)

		if total_sample_number:
			random.seed(1234)
			indices = random.sample(list(range(self.input_imgs.shape[0])), total_sample_number)
			self.input_imgs = self.input_imgs[indices, :, :, :]
			#self.Hy_forward = self.Hy_forward[indices, :, :, :]
			self.fields = self.fields[indices, :, :, :]
			# self.sources = self.sources[indices, :, :, :]

		self.transform = transform

		self.boundary_size = boundary_size
		self.dL = dL
		self.wl = wl

		self.crops_per_idx = crops_per_idx
		self.crop_dx = crop_dx
		self.crop_dy = crop_dy
		# self.cropped_imgs = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, self.crop_dx, self.crop_dy), dtype=np.float32)
		self.cropped_Hys = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, self.crop_dx, self.crop_dy), dtype =np.csingle)
		# self.cropped_sources = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, self.crop_dx, self.crop_dy), dtype =np.csingle)

		self.cropped_yeex = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, self.crop_dx, self.crop_dy), dtype=np.float32)
		self.cropped_yeey = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, self.crop_dx, self.crop_dy), dtype=np.float32)
		self.cropped_top_bc = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, 1, self.crop_dy), dtype =np.csingle)
		self.cropped_bottom_bc = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, 1, self.crop_dy), dtype =np.csingle)
		self.cropped_left_bc = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, self.crop_dx, 1), dtype =np.csingle)
		self.cropped_right_bc = np.zeros((self.input_imgs.shape[0]*self.crops_per_idx, self.crop_dx, 1), dtype =np.csingle)

		self.save_folder = save_folder
		if not os.path.isdir(self.save_folder):
			print("no folder found for path: "+ self.save_folder)
			raise

	def crop_index(self, index):
		assert self.crop_dy <= self.input_imgs.shape[2]
		assert self.crop_dx <= self.input_imgs.shape[1]
		assert self.crop_dx+self.crop_dy+self.crops_per_idx <= self.input_imgs.shape[1] + self.input_imgs.shape[2]
		
		this_image = self.input_imgs[index,:,:]

		this_field_RI = self.fields[index,:,:,:]
		this_field = this_field_RI[:,:,0] + 1j*this_field_RI[:,:,1]

		# this_source_RI = self.sources[index,:,:,:]
		# this_source = 1j*2*np.pi*C_0*self.dL**2/self.wl*EPSILON_0*(this_source_RI[:,:,0] + 1j*this_source_RI[:,:,1])
		# the crop rectangle is with upper left corner (x, y) and bottom right corner (x+w, y+h)
		# x = int((self.input_imgs.shape[1]-self.crop_dx)/2)
		for i in range(self.crops_per_idx):
			x = random.randint(self.boundary_size, self.input_imgs.shape[1]-self.boundary_size-self.crop_dx)
			y = random.randint(self.boundary_size, self.input_imgs.shape[2]-self.boundary_size-self.crop_dy)
			# img_rot0 = this_image[x:x+self.crop_dx, y:y+self.crop_dy]

			# yeex = 1/2*(this_image+np.roll(this_image, 1, axis=0))
			# yeex = yeex[x:x+self.crop_dx, y+1:y+self.crop_dy-1]

			# yeey = 1/2*(this_image+np.roll(this_image, 1, axis=1))
			# yeey = yeey[x+1:x+self.crop_dx-1, y:y+self.crop_dy]

			yeex = 1/2*(this_image+np.roll(this_image, 1, axis=0))
			yeex = yeex[x:x+self.crop_dx, y:y+self.crop_dy]

			yeey = 1/2*(this_image+np.roll(this_image, 1, axis=1))
			yeey = yeey[x:x+self.crop_dx, y:y+self.crop_dy]

			field_rot0 = this_field[x:x+self.crop_dx, y:y+self.crop_dy]
			# source_rot0 = this_source[x:x+self.crop_dx, y:y+self.crop_dy]


			# top_bc0 =    1j*2*np.pi*np.sqrt(np.pad(yeex[1:2, :],  ((0,0),(1,1)), 'constant', constant_values=(0,0)))*self.dL/self.wl*field_rot0[0:1, :] + field_rot0[0:1, :]-field_rot0[1:2, :]
			# bottom_bc0 = 1j*2*np.pi*np.sqrt(np.pad(yeex[-1:, :], ((0,0),(1,1)), 'constant', constant_values=(0,0)))*self.dL/self.wl*field_rot0[-1:, :] + field_rot0[-1:, :]-field_rot0[-2:-1, :]
			# left_bc0 =   1j*2*np.pi*np.sqrt(np.pad(yeey[:, 1:2],  ((1,1), (0,0)), 'constant', constant_values=(0,0)))*self.dL/self.wl*field_rot0[:, 0:1] + field_rot0[:, 0:1]-field_rot0[:, 1:2]
			# right_bc0 =  1j*2*np.pi*np.sqrt(np.pad(yeey[:, -1:], ((1,1), (0,0)), 'constant', constant_values=(0,0)))*self.dL/self.wl*field_rot0[:, -1:] + field_rot0[:, -1:]-field_rot0[:, -2:-1]

			top_bc0 =    1j*2*np.pi*np.sqrt(yeex[1:2, :])*self.dL/self.wl*1/2*(field_rot0[0:1, :]+field_rot0[1:2, :]) + field_rot0[0:1, :]-field_rot0[1:2, :]
			bottom_bc0 = 1j*2*np.pi*np.sqrt(yeex[-1:, :])*self.dL/self.wl*1/2*(field_rot0[-1:, :]+field_rot0[-2:-1, :]) + field_rot0[-1:, :]-field_rot0[-2:-1, :]
			left_bc0 =   1j*2*np.pi*np.sqrt(yeey[:, 1:2])*self.dL/self.wl*1/2*(field_rot0[:, 0:1]+field_rot0[:, 1:2]) + field_rot0[:, 0:1]-field_rot0[:, 1:2]
			right_bc0 =  1j*2*np.pi*np.sqrt(yeey[:, -1:])*self.dL/self.wl*1/2*(field_rot0[:, -1:]+field_rot0[:, -2:-1]) + field_rot0[:, -1:]-field_rot0[:, -2:-1]

			# self.cropped_imgs[(index*self.crops_per_idx+i)] = img_rot0
			self.cropped_Hys[(index*self.crops_per_idx+i)] = field_rot0
			self.cropped_yeex[(index*self.crops_per_idx+i)] = yeex
			self.cropped_yeey[(index*self.crops_per_idx+i)] = yeey
			self.cropped_top_bc[(index*self.crops_per_idx+i)] = top_bc0
			self.cropped_bottom_bc[(index*self.crops_per_idx+i)] = bottom_bc0
			self.cropped_left_bc[(index*self.crops_per_idx+i)] = left_bc0
			self.cropped_right_bc[(index*self.crops_per_idx+i)] = right_bc0
			# self.cropped_sources[(index*self.crops_per_idx+i)] = source_rot0


	def crop_all(self):
		for i in range(self.input_imgs.shape[0]):
			if (i+1) % 10==0: 
				print(i+1)
			self.crop_index(i)
		# np.save(self.save_folder+"cropped_imgs.npy", self.cropped_imgs)
		np.save(self.save_folder+"cropped_Hys.npy", self.cropped_Hys)
		np.save(self.save_folder+"cropped_yeex.npy", self.cropped_yeex)
		np.save(self.save_folder+"cropped_yeey.npy", self.cropped_yeey)
		np.save(self.save_folder+"cropped_top_bc.npy", self.cropped_top_bc)
		np.save(self.save_folder+"cropped_bottom_bc.npy", self.cropped_bottom_bc)
		np.save(self.save_folder+"cropped_left_bc.npy", self.cropped_left_bc)
		np.save(self.save_folder+"cropped_right_bc.npy", self.cropped_right_bc)
		# np.save(self.save_folder+"cropped_sources.npy", self.cropped_sources)


if __name__ == '__main__':
	if len(sys.argv)>1:
		input_dir = sys.argv[1]
		output_dir = sys.argv[2]
		crops_per_idx = int(sys.argv[3])
		ds = SimulationDataset(input_dir,
							   img_file_name="input_eps.npy",
							   field_file_name="Hz_out_forward_RI.npy",
							   sources_file_name="source_RI.npy",
							   crops_per_idx = crops_per_idx, crop_dx=64, crop_dy=64, boundary_size=200, 
							   save_folder=output_dir)

		ds.crop_all()
	else:
		ds = SimulationDataset("/scratch/groups/jonfan/UNet/data/binary_Si_512_512_tilted_500devices",
	                               img_file_name="input_eps.npy",
	                               field_file_name="Hz_out_forward_RI.npy",
	                               sources_file_name="source_RI.npy",
	                               crops_per_idx = 1000, crop_dx=64, crop_dy=64, boundary_size=200, 
	                               save_folder="/scratch/groups/jonfan/UNet/data/Si_500devices_cropped_64_64_robin_no_rotate_500k/")

		ds.crop_all()
