import numpy as np
import dataset_reader

class dataset(object):
	"""This is the dataset class for MNIST"""
	def __init__(self, image_path, label_path):
		super(dataset, self).__init__()
		self.image_path = image_path
		self.label_path = label_path

		self._img_set = dataset_reader.loadImageSet(image_path).astype(np.float32)/255
		self._label_set = dataset_reader.loadLabelSet(label_path).astype(np.float32)
		self._num_instance = self._label_set.shape[0]
				print self._img_set.dtype

	def next_batch(self,batch_size=100, one_hot=True):
		"""
		Get next batch of data
		"""
		order = np.random.permutation(self._num_instance)[:batch_size]
		batch_img = self._img_set[order,:,:,:]	# [img1,img2...]
		batch_label = self._label_set[order,0]	# [l1,l2,...]

		if one_hot:
			new_batch_label = []
			for i in batch_label:
				new_label = np.zeros(10)
				new_label[int(i)] = 1
				new_batch_label.append(new_label)
			del batch_label
			batch_label = np.array(new_batch_label,dtype=np.float32)

		return batch_img, batch_label

	def next_instance(self, one_hot=True):
		"""
		Get next data randomly
		"""
		random_index = np.random.randint(self._num_instance)

		if one_hot:
			label_vec = np.zeros(10)
			label_vec[int(self._label_set[random_index,0])] = 1.0
			return self._img_set[random_index,:,:,:], label_vec

		return self._img_set[random_index,0,:,:], self._label_set[random_index,0]

	def one_hot_labels(self):
		indexs = self._label_set[:,0]
		oh_labels = []
		for i in indexs:
			new_label = np.zeros(10)
			new_label[int(i)] = 1.0
			oh_labels.append(new_label)
		oh_labels = np.array(oh_labels, dtype=np.float32)
		return oh_labels

	@property
	def num_instance(self):
		"""
		Return number of instances in the dataset
		"""
		return self._num_instance

	@property
	def imgs(self):
		"""
		Return all images
		"""
		return self._img_set

	@property
	def labels(self):
		"""
		Return all labels
		"""
		return self._label_set
