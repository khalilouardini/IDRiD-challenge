from __future__ import print_function
from src.metrics import non_max_suppression
from collections import defaultdict, deque
import datetime
import pickle
import time

import torch
import torch.distributed as dist

import errno
import os
import numpy as np


def get_annotations_retinanet(target):
    """Format the target dictionary from the dataset to annotations 
    to be compatible with retinanet"""
    annotations  = np.zeros((0, 5))
    annotation = np.zeros((1, 5))
    for i in range(2):
        
        annotation[0, 0] = target["boxes"][i][0]
        annotation[0, 1] = target["boxes"][i][1]
        annotation[0, 2] = target["boxes"][i][2]
        annotation[0, 3] = target["boxes"][i][3]
        annotation[0, 4]  = i+1
        annotations  = np.append(annotations, annotation, axis=0)

    annotations = np.expand_dims(np.append(annotations, annotation, axis=0), axis=0)
    annotations = torch.from_numpy(annotations).float()
    
    return annotations
    

def get_boxes(model, dataset, threshold = 0.008, img_idx = 0, model_type="FasterRCNN"):
    """method that computes the predicted boxes and filter them using non maximum supression
    Params :
    --------
        model : The model used to make predictions
        dataset
        threshold: used for NMS
        img_idx : img for which we want to extract predicted boxes
        model_type :   "FasterRCNN" or "RetinaNet""
        
    Returns :
    --------
        img : img if index img_idx in dataset
        OD_true_box : ground truth box for OD
        Fovea_true_box : ground truth box for Fovea
        OD_predicted_box : predited OD boxe
        Fovea_predicted_box : prediceted Fovea box after NMS
    
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Select image in test set
    img, target,_ = dataset[img_idx]
    # Put model in evaluation mode
    model.eval()
    # true boxes
    OD_true_box = target["boxes"][0]
    Fovea_true_box = target["boxes"][1]
        
        
    with torch.no_grad(): 
        if model_type =="FasterRCNN":
            prediction = model([img.to(device)])
            boxes = prediction[0]['boxes'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()
            labels = prediction[0]['labels'].cpu().numpy()
        else :
            scores, labels, boxes = model(img.unsqueeze(0).cuda())
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes  = boxes.cpu().numpy()
            
        
    # Retrieve predicted bounding boxes and scores
    
    # retrieve OD box :
    OD_predicted_box = boxes[0]
        
    # retrieve Fovea boxes
    labels = labels[1:]
    scores = scores[1:]
    Fovea_boxes = boxes[1:]
        
    
    # filter predicted boxes 
    if len(Fovea_boxes)>0:   
        kept_idx = list(non_max_suppression(Fovea_boxes, scores, threshold))
        Fovea_boxes = [list(boxes[1:][i]) for i in range(len(boxes[1:])) ]  
        if len(kept_idx)==0 :
            Fovea_predicted_box = boxes[1]
        else :  
            Fovea_predicted_box = Fovea_boxes[kept_idx[0]]
    else :
        print("Fovea boxes empty for img ", img_idx)
        Fovea_predicted_box = boxes[0]
        
    

    #min_dist = np.abs(get_center_distance(min_box, truth_Fovea_box) - mean_distance_dataset)
     
    #for box in Fovea_boxes:
    #    dist = np.abs(get_center_distance(box, truth_Fovea_box) - mean_distance_dataset)
    #    if dist <min_dist:
    #        min_dist = dist
    #        min_box = box
    return img, OD_true_box, Fovea_true_box, OD_predicted_box, Fovea_predicted_box

def get_center(box):
    """return the center of the box"""
    x1,y1,x2,y2 = box
    return [(x1+x2)/2, (y1+y2)/2]
    
def get_center_distance(boxA, boxB, factor=None):
    """returns the distance between two centers multiplied by the scaling factor"""
    centA = get_center(boxA)
    centB = get_center(boxB)
    return np.sqrt( ((1/factor[0])**2) *(centA[0]-centB[0])**2+ ((1/factor[1])**2) *(centA[1]-centB[1])**2 )
        
def get_mean_distance_OD_Fovea(dataset):
    dist = 0
    print("computing mean")
    for i in range(dataset.__len__()):
        boxes = dataset[i][1]['boxes']
        dist += get_center_distance(boxes[0], boxes[1])
    print("done")
    return dist / dataset.__len__()


def compute_means(dataset):
    print("Computing means...")
    return tuple(np.mean([np.mean(dataset[idx][0].numpy(), axis=(1,2)) for idx in range(dataset.__len__())], axis=0))
def compute_stds(dataset):
    print("Computing stds...")
    return tuple(np.std([np.std(dataset[idx][0].numpy(), axis=(1,2)) for idx in  range(dataset.__len__())], axis=0))


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
