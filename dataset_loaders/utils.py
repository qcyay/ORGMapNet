from torchvision.datasets.folder import default_loader
import torch, numpy as np
from torch.nn import functional as F
import cv2, logging, os, os.path as osp, glob, re, collections

old_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda obj: (f'th {tuple(obj.shape)} {obj.type()} '
                                     f'{old_repr(obj)} '
                                     f'type: {obj.type()} shape: {obj.shape} th') if obj.is_contiguous() else (
    f'{tuple(obj.shape)} {obj.type()} '
    f'{old_repr(obj.contiguous())} '
    f'type: {obj.type()} shape: {obj.shape}')
np.set_string_function(lambda arr: f'np {arr.shape} {arr.dtype} '
                                   f'{arr.__str__()} '
                                   f'dtype:{arr.dtype} shape:{arr.shape} np', repr=True)


def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)
    except IOError as e:
        print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    except:
        print('Could not load image {:s}, unexpected error'.format(filename))
        return None
    
    return img


def flatten(t):
    res = []
    for subt in t:
        if isinstance(subt, (list, tuple)):
            res.extend(flatten(subt))
        else:
            res.append(subt)
    return res


def new_collate(batch):
    from torch.utils.data.dataloader import default_collate
    img, meta, pose, pnt = None, None, None, None
    if isinstance(batch[0], dict):
        keys = list(batch[0].keys())
        if 'img' in keys:
            img = [b['img'] for b in batch]
            if img[0] is not None:
                img = default_collate(img)
        if 'pose' in keys:
            pose = default_collate([b['pose'] for b in batch])
        if 'meta' in keys:
            meta = [b['meta'] for b in batch]
            if flatten(meta)[0] is None:
                meta = None
        if 'pnt' in keys:
            pnt = default_collate([b['pnt'] for b in batch])
        return dict(
            img=img, pose=pose, meta=meta, pnt=pnt
        )
    else:
        transposed = list(zip(*batch))
        meta = None
        if len(transposed) == 3:
            img, meta, pose = transposed
        else:
            img, pose = transposed
        if img[0] is not None:
            img = default_collate(img)
        pose = default_collate(pose)
        if flatten(meta)[0] is None:
            meta = None
        res = [img, meta, pose]
        return res


def plt_imshow(img, rsize=False, inp_mode='rgb'):
    from matplotlib import pyplot as plt
    if inp_mode == 'bgr':
        img = img[..., ::-1]
    h, w = img.shape[0], img.shape[1]
    inchh = h / 50
    inchw = w / 50
    if rsize:
        plt.figure(figsize=(inchw, inchh,))
    else:
        plt.figure()
    plt.imshow(img)
    plt.axis('off')


def plt_imshow_tensor(imgs, ncol=4, limit=None):
    import torchvision
    if isinstance(imgs, list):
        imgs = np.asarray(imgs)
    if imgs.shape[-1] == 3:
        imgs = np.transpose(imgs, (0, 3, 1, 2))
    
    imgs_thumb = torchvision.utils.make_grid(
        torch.from_numpy(imgs), normalize=False, scale_each=True,
        nrow=ncol, ).numpy()
    imgs_thumb = to_img(imgs_thumb)
    maxlen = max(imgs_thumb.shape)
    if limit is not None:
        import cvbase as cvb
        imgs_thumb = cvb.resize_keep_ar(imgs_thumb, limit, limit, )
    return imgs_thumb


def rm(path, block=True, remove=False):
    path = osp.abspath(path)
    if not osp.exists(path):
        logging.info(f'no need rm {path}')
    if remove:
        return shell(f'rm -rf "{path}"', block=block)
    stdout, _ = shell('which trash', verbose=False)
    if 'trash' not in stdout:
        dst = glob.glob('{}.bak*'.format(path))
        parsr = re.compile(r'{}.bak(\d+?)'.format(path))
        used = [0, ]
        for d in dst:
            m = re.match(parsr, d)
            if not m:
                used.append(0)
            elif m.groups()[0] == '':
                used.append(0)
            else:
                used.append(int(m.groups()[0]))
        dst_path = '{}.bak{}'.format(path, max(used) + 1)
        cmd = 'mv {} {} '.format(path, dst_path)
        return shell(cmd, block=block)
    else:
        return shell(f'trash -r "{path}"', block=block)


def mkdir_p(path, delete=False, verbose=True):
    import os, os.path as osp
    
    path = str(path)
    if path == '':
        return
    if delete and osp.exists(path):
        rm(path)
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)


def shell(cmd, block=True, return_msg=True, verbose=True, timeout=None):
    import os, logging, subprocess
    my_env = os.environ.copy()
    home = os.path.expanduser('~')
    my_env['PATH'] = home + "/anaconda3/bin/:" + my_env['PATH']
    my_env['http_proxy'] = ''
    my_env['https_proxy'] = ''
    if verbose:
        logging.info('cmd is ' + cmd)
    if block:
        # subprocess.call(cmd.split())
        task = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=my_env,
                                preexec_fn=os.setsid
                                )
        if return_msg:
            msg = task.communicate(timeout)
            msg = [msg_.decode('utf-8') for msg_ in msg]
            if msg[0] != '' and verbose:
                logging.info('stdout {}'.format(msg[0]))
            if msg[1] != '' and verbose:
                logging.error(f'stderr {msg[1]}, cmd {cmd}')
            return msg
        else:
            return task
    else:
        logging.debug('Non-block!')
        task = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=my_env,
                                preexec_fn=os.setsid
                                )
        return task


def to_numpy(tensor):
    import PIL
    if isinstance(tensor, torch.autograd.Variable):
        tensor = tensor.detach()
    if torch.is_tensor(tensor):
        if tensor.shape == ():
            tensor = tensor.item()
            tensor = np.asarray([tensor])
        elif np.prod(tensor.shape) == 1:
            tensor = tensor.item()
            tensor = np.asarray([tensor])
        else:
            tensor = tensor.cpu().numpy()
            tensor = np.asarray(tensor)
    if type(tensor).__module__ == 'PIL.Image':
        tensor = np.asarray(tensor)
    # elif type(tensor).__module__ != 'numpy':
    #     raise ValueError("Cannot convert {} to numpy array"
    #                      .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if ndarray is None:
        return None
    if isinstance(ndarray, collections.Sequence):
        return [to_torch(ndarray_) for ndarray_ in ndarray if ndarray_ is not None]
    # if isinstance(ndarray, torch.autograd.Variable):
    #     ndarray = ndarray.data
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def to_img(img, target_shape=None):
    if isinstance(img, str):
        assert osp.exists(img), img
        img = cv2.imread(img)[..., ::-1]
    img = np.array(img)
    img = img.copy()
    shape = img.shape
    if len(shape) == 3 and shape[-1] == 4:
        img = img[..., :3]
    if len(shape) == 3 and shape[0] == 3:
        img = img.transpose(1, 2, 0)
        img = np.array(img, order='C')
    # if img.dtype == np.float32 or img.dtype == np.float64:
    img -= img.min()
    img = img / (img.max() + 1e-6)
    img *= 255
    img = np.array(img, dtype=np.uint8)
    if len(shape) == 3 and shape[-1] == 1:
        img = img[..., 0]
    if target_shape:
        # img = np.uint8(Image.fromarray(img).resize(target_shape, Image.ANTIALIAS)) # 128,256
        img = img.astype('float32')
        img = to_torch(img).unsqueeze(0).unsqueeze(0)
        img = F.interpolate(img, size=target_shape, mode='bilinear', align_corners=True)
        img = img.squeeze(0).squeeze(0)
        img = to_numpy(img).astype('uint8')
    return img.copy()


def trans_bbox_cv2mat_fmt(bbox):
    # this is x,y in opencv format
    cx, cy, w, h, scr = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3], bbox[:, 4]
    return np.stack((cy, cx, h, w, scr), axis=1)


def trans_bbox_mat2cv2_fmt(bbox):
    # in fact we can reuse the func
    return trans_bbox_cv2mat_fmt(bbox)


def trans_bbox_center2pnts(bbox):
    cx, cy, w, h, scr = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3], bbox[:, 4]
    hw = w / 2;
    hh = h / 2;
    left_top = (cx - hw, cy - hh)
    right_bottom = (cx + hw, cy + hh)
    return np.stack(
        left_top + right_bottom + (scr,),
        axis=1
    )


def draw_bbox(img_test, dets_test, scr_thresh=.3, text=None):
    for idx, bb in enumerate(dets_test):
        x1, y1, x2, y2, scr = bb
        if scr > scr_thresh:
            img_test = cv2.rectangle(img_test, (x1, y1), (x2, y2), (0, 255, 0), 1)
            if text is not None:
                rat = .7
                nrat = 1 - rat
                img_test = cv2.putText(img_test,
                                       text[idx], (int(x1), int(y1 + 11)),
                                       cv2.FONT_HERSHEY_SIMPLEX, .5,
                                       (255, 255, 0), 1, cv2.LINE_AA
                                       )
    return img_test


def resize_bbox(bboxs, ori, tgt):
    owid, ohei = ori
    lwid, lhei = tgt
    ratw, rath = lwid / owid, lhei / ohei
    col, row, wid, hei, scores = bboxs[:, 0], bboxs[:, 1], bboxs[:, 2], bboxs[:, 3], bboxs[:, 4]
    col *= ratw
    wid *= ratw
    row *= rath
    hei *= rath
    return np.stack((col, row, wid, hei, scores), axis=1)


def fd2flst(image_folder, down=None):
    images = [img for img in os.listdir(image_folder)]
    images = list(filter(lambda x: 'depth' not in x, images))
    images = list(filter(lambda x: 'png' in x or 'jpg' in x, images))
    images = sorted(images)
    images = sorted(images, key=len)
    if down:
        images = images[::down]
    return images


def fd2vdo(image_folder, video_name='video.mp4', fps=5, down=None):
    import cv2
    import os
    images = fd2flst(image_folder, down)
    fn = os.path.join(image_folder, images[0])
    assert osp.exists(fn)
    frame = cv2.imread(fn)
    assert frame is not None, fn
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'),
                            frameSize=(width, height), fps=fps)
    if down:
        images = images[::3]
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    
    cv2.destroyAllWindows()
    video.release()


import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2


# 将像素点从图像平面投影到空间坐标系
# 输入pts_depth，像素点的位置和深度，前两列为像素点的位置，第三列为像素点的深度，尺寸为[N,3]
# 输出pts_3d_rect，像素点在空间中的位置，尺寸为[N,3]
def project_img_to_rect(pts_depth,
                        fx, fy, cx, cy, factor=5000,
                        ):
    n = pts_depth.shape[0]
    # 尺寸为[N]
    z = pts_depth[:, 2] / factor
    # 尺寸为[N]
    x = ((pts_depth[:, 1] - cx) * z) / fx
    # 尺寸为[N]
    y = ((pts_depth[:, 0] - cy) * z) / fy
    # 尺寸为[N,3]
    pts_3d_rect = np.zeros((n, 3))
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = z
    
    return pts_3d_rect


# RT矩阵获取函数
def RT_metric(t, q):
    r = R.from_quat(q)
    r = r.as_matrix()
    RT = np.zeros((4, 4))
    RT[:3, :3] = r
    RT[:3, 3] = t
    RT[3, 3] = 1
    return RT


def relative_RT(R1, R2):
    R2 = np.linalg.inv(R2)
    R21 = np.dot(R2, R1)
    return R21


# 将像素点从空间坐标系投影到图像平面
# 输入pts_rect，像素点在空间中的位置，尺寸为[N,3]，K，内参矩阵
# 输出pts_2d，像素点的位置，尺寸为[N,2]
def project_rect_to_img(pts_rect, K):
    pts_rect = np.insert(pts_rect, 3, 1, 1)
    P = np.insert(K, 3, 0, 1)
    pts_2d = np.dot(pts_rect, P.T)
    pts_2d[:, 0] = pts_2d[:, 0] / pts_2d[:, 2]
    pts_2d[:, 1] = pts_2d[:, 1] / pts_2d[:, 2]
    pts_2d = pts_2d[:, :2]
    pts_2d = pts_2d[:, ::-1]
    return pts_2d

def permutation_generator(bbox, n):

    N = bbox.shape[0]
    n = np.random.randint(0, n+1)
    m = N - n
    if m < 0:
        m = 0
    perm = torch.randperm(N)[:m]

    return perm
