"""Utils"""
import skvideo.io
import os
import glob

from numpy import nanmean
from tqdm import tqdm
import ffmpeg
import torch

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sample_video_tuples(video_fname,  output_dir, clip_length=16, interval_length=8, tuple_length=3):
	videodata = skvideo.io.vread(video_fname)
	videoname = os.path.basename(video_fname)
	videoname = videoname[:videoname.rfind('.')]

	clip_start = 0
	for i in range(tuple_length):
		clipname = os.path.join(output_dir, '{}_{}.avi'.format(videoname, i+1))
		clipdata = videodata[clip_start:clip_start+clip_length]
		clip_start = clip_start+clip_length+interval_length
		skvideo.io.vwrite(clipname, clipdata)

def skvideo_io_vread_compatiable(dir):
	filenames = glob.glob(os.path.join(dir, '**/*.avi'), recursive=True)
	for filename in tqdm(filenames):
		dirname, basename = os.path.split(filename)
		tmp_filename = os.path.join(dirname, 'tmp.avi')
		try:
			videodata = skvideo.io.vread(filename)
		except Exception:
			print('Convert ', filename)
			ffmpeg.input(filename).output(tmp_filename).run(quiet=True)
			os.remove(filename)
			os.rename(tmp_filename, filename)


def calculate_accuracy(outputs, targets):
	batch_size = targets.size(0)

	_, pred = outputs.topk(1, 1, True)
	pred = pred.t()
	correct = pred.eq(targets.view(1, -1))
	n_correct_elems = correct.float().sum().data[0]

	return n_correct_elems / batch_size


def calculate_accuracy_per_class(outputs, targets, class_idx):
	targets = targets[targets == class_idx]
	outputs = outputs[targets == class_idx]

	batch_size = targets.size(0)
	_, pred = outputs.topk(1, 1, True)
	pred = pred.t()
	correct = pred.eq(targets.view(1, -1))
	n_correct_elems = correct.float().sum().data[0]

	return n_correct_elems / batch_size


class BinaryClassificationMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.tp = 0
		self.tn = 0
		self.fp = 0
		self.fn = 0
		self.acc = 0
		self.pre = 0
		self.rec = 0
		self.f1 = 0
		self.val = 0

	def update(self, output, target):
		pred = output >= 0.5
		truth = target >= 0.5
		tp = pred.mul(truth).sum(0).float()
		tn = (pred.bitwise_not()).mul(truth.bitwise_not()).sum(0).float()
		fp = pred.mul(1 - truth).sum(0).float()
		fn = (1 - pred).mul(truth).sum(0).float()
		self.val = (tp + tn).sum() / (tp + tn + fp + fn).sum()
		self.tp += tp
		self.tn += tn
		self.fp += fp
		self.fn += fn
		self.acc = (self.tp + self.tn).sum() / (self.tp + self.tn + self.fp + self.fn).sum()
		self.pre = self.tp / (self.tp + self.fp)
		self.rec = self.tp / (self.tp + self.fn)
		self.f1 = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn)
		self.avg_pre = nanmean(self.pre)
		self.avg_rec = nanmean(self.rec)
		self.avg_f1 = nanmean(self.f1)


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


if __name__ == "__main__":
	# sample_video_tuples('data/ucf101/video/BasketballDunk/v_BasketballDunk_g01_c01.avi', '.')
	skvideo_io_vread_compatiable('data/ucf101/video')