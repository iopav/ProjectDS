import os
"""
保存和加载模型检查点的实用函数。
函数:
	save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir='checkpoints', max_checkpoints=5):
		保存模型、优化器和调度器的状态到指定目录的检查点文件中。
		参数:
			model (torch.nn.Module): 要保存的模型。
			optimizer (torch.optim.Optimizer): 要保存的优化器。
			scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 要保存的学习率调度器。
			epoch (int): 当前的训练轮次。
			checkpoint_dir (str, optional): 保存检查点的目录。默认值为 'checkpoints'。
			max_checkpoints (int, optional): 保留的最大检查点数量。默认值为 5。
	load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
		从检查点文件中加载模型、优化器和调度器的状态。
		参数:
			checkpoint_path (str): 检查点文件的路径。
			model (torch.nn.Module): 要加载状态的模型。
			optimizer (torch.optim.Optimizer, optional): 要加载状态的优化器。
			scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 要加载状态的学习率调度器。
		返回:
			model (torch.nn.Module): 加载了状态的模型。
			optimizer (torch.optim.Optimizer, optional): 加载了状态的优化器。
			scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 加载了状态的学习率调度器。
			epoch (int): 检查点保存时的训练轮次。
	latest_checkpoint_path(dir_path, regex="checkpoint_*.pth"):
		获取指定目录中最新的检查点文件路径。
		参数:
			dir_path (str): 检查点文件所在的目录。
			regex (str, optional): 用于匹配检查点文件的正则表达式。默认值为 "checkpoint_*.pth"。
		返回:
			str: 最新的检查点文件路径。
"""
import torch
import logging
import glob
import random
import torchvision.transforms.functional as F



# 使用示例
# input_csv = "triplet.csv"  # 输入文件路径
# output_csv = "pair.csv"    # 输出文件路径
# convert_triplet_to_pair(input_csv, output_csv)



def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir='checkpoints', max_checkpoints=5):
	# 如果检查点目录不存在，则创建该目录
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	# 定义检查点文件的路径
	checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth')
	scheduler_state = None
	# 如果调度器不为空，则获取调度器的状态字典
	if scheduler is not None:
		scheduler_state = scheduler.state_dict()

	# 保存模型、优化器和调度器的状态到检查点文件中
	torch.save({
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'scheduler': scheduler_state,
		'epoch': epoch
	}, checkpoint_path)

	# 打印保存检查点的信息
	print(f"Saving model and optimizer state at epoch {epoch} to {checkpoint_path}")
	# 获取所有检查点文件并按训练轮次排序
	checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')], key=lambda x: int(x.split('_')[1].split('.')[0]))
	# 如果检查点数量超过最大限制，则删除最旧的检查点
	if len(checkpoints) > max_checkpoints:
		for ckpt in checkpoints[:-max_checkpoints]:
			os.remove(os.path.join(checkpoint_dir, ckpt))

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
	# 确认检查点文件存在
	assert os.path.isfile(checkpoint_path)
	# 加载检查点文件
	checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
	epoch = checkpoint_dict['epoch']
	# 如果调度器不为空，则加载调度器的状态
	if scheduler is not None:
		scheduler.load_state_dict(checkpoint_dict['scheduler'])
	# 如果优化器不为空，则加载优化器的状态
	if optimizer is not None:
		optimizer.load_state_dict(checkpoint_dict['optimizer'])
	saved_state_dict = checkpoint_dict['model']
	state_dict = model.state_dict()
	new_state_dict = {}
	# 加载模型的状态字典
	for k, v in state_dict.items():
		try:
			new_state_dict[k] = saved_state_dict[k]
		except:
			new_state_dict[k] = v
	model.load_state_dict(new_state_dict)
	# 打印加载检查点的信息
	print(f"Loaded checkpoint '{checkpoint_path}' (epoch {epoch})")
	return model, optimizer, scheduler, epoch

def latest_checkpoint_path(dir_path, regex="checkpoint_*.pth"):
	# 获取指定目录中所有匹配正则表达式的检查点文件
	f_list = glob.glob(os.path.join(dir_path, regex))
	# 按文件名中的数字部分排序
	f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
	# 返回最新的检查点文件路径
	x = f_list[-1]
	return x
