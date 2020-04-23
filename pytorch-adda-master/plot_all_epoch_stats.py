# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:43:17 2020

@author: Jinsung
"""

import numpy as np

def prune_ticks_labels(ticks, labels):
	ticks = np.asarray(ticks)
	labels = np.asarray(labels)

	if len(labels) > 15 and len(labels) <= 50:
		idx = np.where(labels % 5 == 0)
		labels = labels[idx]
		ticks = ticks[idx]
	elif len(labels) > 50 and len(labels) <= 100:
		idx = np.where(labels % 10 == 0)
		labels = labels[idx]
		ticks = ticks[idx]
	elif len(labels) > 100:
		idx = np.where(labels % 25 == 0)
		labels = labels[idx]
		ticks = ticks[idx]
	return ticks, labels

def parse_all_epoch_stats(all_epoch_stats, prune=True):
	base_iter_count = 0
	ticks = []
	labels = []
	xs = []
	acc = []
	err = []

	for epoch, epoch_stats in enumerate(all_epoch_stats):
		for stats in epoch_stats:
			err.append(stats[2])
			acc.append(stats[3])
			xs.append(base_iter_count + stats[0])
			base_iter_count += stats[1]
		ticks.append(base_iter_count)
		labels.append(epoch+1)

	if prune:
		ticks, labels = prune_ticks_labels(ticks, labels)
	return ticks, labels, xs, err, acc

def plot_all_epoch_stats(all_epoch_stats, outf):
	import matplotlib.pyplot as plt
	ticks, labels, xs, err, acc = parse_all_epoch_stats(all_epoch_stats)

	plt.plot(xs, np.asarray(acc) * 100, color='k', label='ACC')
	#plt.plot(xs, np.asarray(tg_te_err)*100, color='r', label='target')
	#plt.plot(xs, np.asarray(sc_te_err)*100, color='b', label='source')
	plt.plot(xs, np.asarray(err), color='y', label='loss')

	colors = ['g', 'm', 'c', 'y']
	'''
	for i in range(us_te_err.shape[1]):
		plt.plot(xs, np.asarray(us_te_err[:,i])*100, color=colors[i], label='self-sup %d' %(i+1))
	'''
	plt.xticks(ticks, labels)
	plt.xlabel('epoch')
	plt.ylabel('test error (%)')
	plt.legend()
	plt.savefig('%s/loss.pdf' %(outf))
	plt.close()